"""Composable loss function modules operating on state PyTrees.

TODO:

- The time aggregation could be done in `CompositeLoss`, if we unsqueeze
  terms that don't have a time dimension. This would allow time aggregation
  to be controlled in one place, if for some reason it makes sense to change
  how this aggregation occurs across all loss terms.
  - Actually I think the time aggregation should probably be left to the
    specific implementation; consider that we might want to do dynamic slicing
    and aggregation which involves vmapping over a function in which the time
    aggregation occurs.
- Protocols for all the different `state` types/fields?
    - Alternatively we could make `AbstractLoss` generic over a
      `StateT` typevar, however that might not make sense for typing
      the compositions (e.g. `__sum__`) since the composite can support any
      state pytrees that have the right combination of fields, not just pytrees
      that have an identical structure.

:copyright: Copyright 2023-2024 by Matt Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

#! Can't do this because `AbstractVar` annotations can't be stringified.
# from __future__ import annotations

import functools as ft
import inspect
import logging
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, partial
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Optional,
    Self,
    Tuple,
    TypeVar,
    Union,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import jax.tree_util as jtu
from equinox import AbstractVar, Module, field
from jax_cookbook.misc import moving_avg, softmin
from jaxtyping import Array, ArrayLike, Float, PyTree

from feedbax._mapping import WhereDict
from feedbax._model import AbstractModel
from feedbax.misc import get_unique_label, unzip2
from feedbax.state import State

if TYPE_CHECKING:
    from feedbax.bodies import SimpleFeedbackState
    from feedbax.task import TaskTrialSpec


logger = logging.getLogger(__name__)


U = TypeVar("U")


@jtu.register_pytree_node_class
@dataclass
class TermTree[T](Mapping[str, "TermTree"]):
    label: str
    names: Tuple[str, ...]
    children: Tuple[Self, ...] = ()
    value: Optional[Array] = None  # only for leaf
    weight: float = 1.0
    leaf_fn: Callable[[Array], Array] = jnp.mean
    originator: Optional[T] = None

    def tree_flatten(self):
        """Flatten for PyTree; only include numeric/JAX parts dynamically."""
        # dynamic leaves (these are mapped by vmap/jit):
        children = (self.children, self.value)
        # static metadata (these are carried in aux data, not mapped):
        aux = (self.label, self.names, self.weight, self.leaf_fn, self.originator)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        label, names, weight, leaf_fn, originator = aux_data
        kids, value = children
        return cls(
            label=label,
            names=names,
            children=kids,
            value=value,
            weight=weight,
            leaf_fn=leaf_fn,
            originator=originator,
        )

    @staticmethod
    def leaf(
        label: str,
        value: Array,
        *,
        weight: float = 1.0,
        leaf_fn: Callable[[Array], Array] = jnp.mean,
        originator: Optional[T] = None,
    ) -> "TermTree":
        return TermTree(
            label=label,
            names=(),
            children=(),
            value=value,
            leaf_fn=leaf_fn,
            weight=weight,  # originator=originator
        )

    @staticmethod
    def branch(
        label: str,
        mapping: Mapping[str, "TermTree"],
        *,
        weight: float = 1.0,
        leaf_fn: Callable[[Array], Array] = jnp.mean,
        originator: Optional[T] = None,  #! I don't think this should use the class generic T
    ) -> "TermTree":
        # Freeze order for JIT-stability
        names = tuple(mapping.keys())
        kids = tuple(mapping[k] for k in names)
        return TermTree(
            label=label,
            names=names,
            children=kids,
            value=None,
            weight=weight,
            leaf_fn=leaf_fn,
            # originator=originator,
        )

    def with_weight(self, weight: float) -> Self:
        return TermTree(
            label=self.label,
            names=self.names,
            children=self.children,
            value=self.value,
            leaf_fn=self.leaf_fn,
            weight=weight,
        )

    def fold(
        self,
        *,
        on_leaf: Callable[["TermTree"], U],
        on_branch: Callable[["TermTree", Tuple[U, ...]], U],
    ) -> U:
        if self.value is not None:
            return on_leaf(self)
        child_vals = tuple(c.fold(on_leaf=on_leaf, on_branch=on_branch) for c in self.children)
        return on_branch(self, child_vals)

    def aggregate(
        self,
        *,
        leaf_fn: Optional[
            Callable[[Array], Array]
        ] = None,  # uniform override; None → use node.leaf_fn
        plus: Callable[[Array, Array], Array] = jnp.add,  # how to combine siblings
        times: Callable[[Array, Array], Array] = jnp.multiply,  # how to apply weights
        zero: Array | float = 0.0,  # identity for `plus`
    ) -> Array:
        # (a) how to evaluate a leaf:
        def _on_leaf(node: "TermTree") -> Array:
            f = node.leaf_fn if leaf_fn is None else leaf_fn
            v = f(node.value)  # scalar (or broadcastable) is expected
            return times(jnp.asarray(node.weight), v)

        # (b) how to evaluate a branch:
        def _on_branch(node: "TermTree", kids: Tuple[Array, ...]) -> Array:
            acc = ft.reduce(plus, kids, jnp.asarray(zero))
            return times(jnp.asarray(node.weight), acc)

        return self.fold(on_leaf=_on_leaf, on_branch=_on_branch)

    @property
    def total(self) -> Array:
        """Return the node-weighted scalar sum of all leaves in this tree."""
        return self.aggregate()

    def map(self, fn: Callable[[Array], ArrayLike], check_value: Callable = eqx.is_array) -> Self:
        """Apply a function to all `value` arrays in this `TermTree`.

        Handle the recursion explicitly rather than using `jt.map` and excluding `weights` etc.
        """
        if self.value is not None:
            # Leaf node
            new_value = fn(self.value) if check_value(self.value) else self.value
            return eqx.tree_at(lambda t: t.value, self, new_value)
        else:
            # Branch node - recursively apply to children
            new_children = tuple(child.map(fn) for child in self.children)
            return eqx.tree_at(lambda t: t.children, self, new_children)

    def _walk_leaves(
        self,
        *,
        include_root: bool,
        apply_weights: bool,
    ):
        """Yield (path:str, node:TermTree, cum_w:float)."""

        def visit(node: "TermTree", path_parts: list[str], cum_w: float):
            if node.value is not None:
                # leaf
                path = "/".join(path_parts) if path_parts else node.label
                yield path, node, (cum_w * node.weight if apply_weights else 1.0)
                return

            next_cum_w = cum_w * (node.weight if apply_weights else 1.0)
            base_parts = [node.label] if include_root and not path_parts else path_parts
            for name, child in zip(node.names, node.children):
                yield from visit(child, base_parts + [name], next_cum_w)

        yield from visit(self, [], 1.0)

    def flatten(
        self,
        *,
        apply_weights: bool = True,
        apply_leaf_fn: bool = False,
        include_root: bool = False,
    ) -> dict[str, Array]:
        """
        Returns {'path/to/leaf': Array} for all leaves, WITHOUT applying `node.leaf_fn`.

        If `apply_weights=True`, multiplies the raw leaf arrays by the cumulative
        product of weights along the path (including the leaf's own `weight`).
        """
        out: dict[str, Array] = {}
        for path, node, w in self._walk_leaves(
            include_root=include_root, apply_weights=apply_weights
        ):
            #! TODO: Make this `dict[str, Array | None]` and keep `None` leaves
            arr = node.value if node.value is not None else jnp.asarray(0.0)
            arr = self.leaf_fn(arr) if apply_leaf_fn else arr
            out[path] = (w * arr) if apply_weights else arr
        return out

    def iter_items(self, *, include_root: bool = False):
        """Yield (path, leaf_node) for all leaves (no weights, no reduction)."""
        for path, node, _ in self._walk_leaves(include_root=include_root, apply_weights=False):
            if node.value is not None:
                yield path, node

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return iter(self.names)

    def __getitem__(self, k: str) -> Self:
        i = self.names.index(k)
        return self.children[i]


def is_termtree(x):
    return isinstance(x, TermTree)


class AbstractLoss(Module):
    """Abstract base class for loss functions.

    Instances can be composed by addition and scalar multiplication.

    For leaf loss terms, subclass this and implement the `term` method.

    For nodes composed of multiple weighted terms, instantiate `CompositeLoss`.

    For a composite leaf that includes multiple weighted sub-terms that all
    depend on shared variables, instantiate `FuncTermsLoss`.
    """

    label: AbstractVar[str]

    def __call__(
        self,
        states: PyTree,
        trial_specs: "TaskTrialSpec",
        model: AbstractModel,
    ) -> TermTree["AbstractLoss"]:
        return TermTree.leaf(
            self.label,
            self.term(states, trial_specs, model),
            originator=self,
        )

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        """Implement this to calculate a loss term."""
        raise NotImplementedError

    # TODO
    # def __init_subclass__(cls, **kwargs):
    #     """Enforce that subclasses implement exactly one of `term` (leaf nodes) or `__call__`."""
    #     super().__init_subclass__(**kwargs)
    #     if cls is AbstractLoss:
    #         return  # don't check the root

    #     base_call = AbstractLoss.__call__
    #     base_term = AbstractLoss.term

    #     # effective implementations after MRO resolution
    #     call_impl = getattr(cls, "__call__", None)
    #     term_impl = getattr(cls, "term", None)

    #     call_overridden = call_impl is not base_call
    #     term_overridden = term_impl is not base_term

    #     # allow abstract intermediates that implement neither
    #     if not call_overridden and not term_overridden:
    #         #! Doesn't work, even if we subclass ABC; need to check if abstract `eqx.Module`
    #         if not inspect.isabstract(cls):
    #             raise TypeError(
    #                 f"{cls.__name__} must override exactly one of '__call__' or 'term'."
    #             )
    #         return

    #     # forbid overriding both
    #     if call_overridden and term_overridden:
    #         raise TypeError(
    #             f"{cls.__name__} must override exactly one of '__call__' or 'term', not both."
    #         )

    def skeleton(self, batch_dims: tuple[int, ...]) -> TermTree:
        return TermTree.leaf(self.label, jnp.empty(batch_dims))

    def __add__(self, other: "AbstractLoss") -> "CompositeLoss":
        return CompositeLoss(terms=(self, other), weights=(1.0, 1.0))

    def __radd__(self, other: "AbstractLoss") -> "CompositeLoss":
        return self.__add__(other)

    def __sub__(self, other: "AbstractLoss") -> "CompositeLoss":
        # ? I don't know if this even makes sense but it's easy to implement.
        return CompositeLoss(terms=(self, other), weights=(1.0, -1.0))

    def __rsub__(self, other: "AbstractLoss") -> "CompositeLoss":
        return CompositeLoss(terms=(self, other), weights=(-1.0, 1.0))

    def __neg__(self) -> "CompositeLoss":
        return CompositeLoss(terms=(self,), weights=(-1.0,))

    def __mul__(self, other) -> "CompositeLoss":
        """Assume scalar multiplication."""
        if eqx.is_array_like(other):
            if eqx.is_array(other) and not other.shape == ():
                raise ValueError("Can't multiply loss term by non-scalar array")
            return CompositeLoss(terms=(self,), weights=(other,))
        else:
            raise ValueError("Can't multiply loss term by non-numeric type")

    def __rmul__(self, other):
        return self.__mul__(other)


class AbstractTermedLoss(AbstractLoss):
    terms: AbstractVar[Mapping[str, Any]]
    weights: AbstractVar[Mapping[str, float]]

    def flatten_weights(
        self,
        *,
        prefix: str = "",
        apply_weights: bool = True,
        parent_weight: float = 1.0,
        include_self_label: bool = False,
    ) -> dict[str, float]:
        """
        Return a flattened mapping { 'path/to/leaf': effective_weight }.

        If apply_weights=True, each value includes the product of all weights
        along the path from the root down to that leaf. Otherwise, only the
        local (child) weights are returned.
        """
        base = prefix
        if include_self_label:
            base += self.label
        out: dict[str, float] = {}
        if base:
            base += "/"

        for name, subloss in self.terms.items():
            local_w = self.weights.get(name, 1.0)
            path_prefix = f"{base}{name}"

            if isinstance(subloss, AbstractTermedLoss):
                # Recurse into sub-composites
                sub = subloss.flatten_weights(
                    prefix=path_prefix,
                    apply_weights=apply_weights,
                    parent_weight=(parent_weight * local_w if apply_weights else 1.0),
                )
                out.update(sub)
            else:
                # Leaf: record effective weight
                effective = parent_weight * local_w if apply_weights else local_w
                out[path_prefix] = float(effective)

        return out

    def skeleton(self, batch_dims: tuple[int, ...]) -> TermTree:
        children: dict[str, TermTree] = {}
        for name, term in self.terms.items():
            if isinstance(term, AbstractLoss):
                child = term.skeleton(batch_dims)
            else:
                child = TermTree.leaf(name, jnp.empty(batch_dims))
            child = child.with_weight(self.weights.get(name, 1.0))
            children[name] = child
        return TermTree.branch(self.label, children, originator=self)


class FuncTermsLoss[T](AbstractTermedLoss):
    """A leaf loss node with multiple weighted terms that depend on shared context."""

    label: str
    build_context: Callable[
        [State, "TaskTrialSpec", AbstractModel], T
    ]  # (states, trial_specs, model) -> Ctx
    terms: Mapping[str, Callable[[T], Array]]
    weights: Mapping[str, float]

    @jax.named_scope("fbx.FuncTermsLoss")
    def __call__(
        self,
        states: State,
        trial_specs: "TaskTrialSpec",
        model: AbstractModel,
    ) -> TermTree[AbstractLoss]:
        ctx = self.build_context(states, trial_specs, model)
        children = {}
        for name, fn in self.terms.items():
            v = fn(ctx)
            # v = v if v.shape == () else self.reduce(v)  # enforce scalar per component
            leaf = TermTree.leaf(name, v).with_weight(self.weights.get(name, 1.0))
            children[name] = leaf
        return TermTree.branch(self.label, children, originator=self)


class CompositeLoss(AbstractTermedLoss):
    """A loss node that composes multiple loss nodes and their weights."""

    label: str
    terms: dict[str, AbstractLoss]
    weights: dict[str, float]

    def __init__(
        self,
        terms: Mapping[str, AbstractLoss] | Sequence[AbstractLoss],
        weights: Optional[Mapping[str, float] | Sequence[float]] = None,
        label: str = "",
        user_labels: bool = True,
    ):
        """
        !!! Note
            During construction the user may pass dictionaries and/or sequences
            of `AbstractLoss` instances (`terms`) and weights.

            Any `CompositeLoss` instances in `terms` are flattened, and their
            simple terms incorporated directly into the new composite loss,
            with the weights of those simple terms multiplied by the weight
            given in `weights` for their parent composite term.

            If a composite term has a user-specified label, that label will be
            prepended to the labels of its component terms, on flattening. If
            the flattened terms still do not have unique labels, they will be
            suffixed with the lowest integer that makes them unique.

        Arguments:
            terms: The sequence or mapping of loss terms to be included.
            weights: A float PyTree of the same structure as `terms`, giving
                the scalar term weights. By default, all terms have equal weight.
            label: The label for the composite loss.
            user_labels: If `True`, the keys in `terms`---if it is a mapping---
                are used as term labels, instead of the `label` field of each term.
                This is useful because it may be convenient for the user to match up
                the structure of `terms` and `weights` in a PyTree such as a dict,
                which provides labels, yet continue to use the default labels.
        """
        self.label = label

        if isinstance(terms, Mapping):
            if user_labels:
                labels, terms = list(zip(*terms.items()))
            else:
                labels = [term.label for term in terms.values()]
                terms = list(terms.values())
        elif isinstance(terms, Sequence):
            # TODO: if `terms` is a dict, this fails!
            labels = [term.label for term in terms]
        else:
            raise ValueError("terms must be a mapping or sequence of AbstractLoss")

        if isinstance(weights, Mapping):
            weight_values = tuple(weights.values())
        elif isinstance(weights, Sequence):
            weight_values = tuple(weights)
        elif weights is None:
            weight_values = tuple(1.0 for _ in terms)

        if not len(terms) == len(weight_values):
            raise ValueError("Mismatch between number of loss terms and number of term weights")

        # Split into lists of data for simple and composite terms.
        term_tuples_split: Tuple[
            Sequence[Tuple[str, AbstractLoss, float]],
            Sequence[Tuple[str, AbstractLoss, float]],
        ]
        term_tuples_split = eqx.partition(
            list(zip(labels, terms, weight_values)),
            lambda x: not isinstance(x[1], CompositeLoss),
            is_leaf=lambda x: isinstance(x, tuple),
        )

        # Removes the `None` values from the lists.
        term_tuples_leaves = jt.map(
            lambda x: jtu.tree_leaves(x, is_leaf=lambda x: isinstance(x, tuple)),
            term_tuples_split,
            is_leaf=lambda x: isinstance(x, list),
        )

        # Start with the simple terms, if there are any.
        if term_tuples_leaves[0] == []:
            all_labels, all_terms, all_weights = (), (), ()
        else:
            all_labels, all_terms, all_weights = zip(*term_tuples_leaves[0])

        # Make sure the simple term labels are unique.
        for i, label in enumerate(all_labels):
            label = get_unique_label(label, all_labels[:i])
            all_labels = all_labels[:i] + (label,) + all_labels[i + 1 :]

        # Flatten the composite terms, assuming they have the usual dict
        # attributes. We only need to flatten one level, because this `__init__`
        # (and the immutability of `eqx.Module`) ensures no deeper nestings
        # are ever constructed except through extreme hacks.
        for group_label, composite_term, group_weight in term_tuples_leaves[1]:
            labels = composite_term.terms.keys()

            # If a unique label for the composite term is available, use it to
            # format the labels of the flattened terms.
            if group_label != "":
                labels = [f"{group_label}_{label}" for label in labels]
            elif composite_term.label != "":
                labels = [f"{composite_term.label}_{label}" for label in labels]

            # Make sure the labels are unique.
            for label in labels:
                label = get_unique_label(label, all_labels)
                all_labels += (label,)

            all_terms += tuple(composite_term.terms.values())
            all_weights += tuple(
                [group_weight * weight for weight in composite_term.weights.values()]
            )

        self.terms = dict(zip(all_labels, all_terms))
        self.weights = dict(zip(all_labels, all_weights))

    def __or__(self, other: "CompositeLoss") -> "CompositeLoss":
        """Merge two composite losses, overriding terms with the same label."""
        return CompositeLoss(
            terms=self.terms | other.terms,
            weights=self.weights | other.weights,
            label=other.label,
        )

    @jax.named_scope("fbx.CompositeLoss")
    def __call__(
        self,
        states: State,
        trial_specs: "TaskTrialSpec",
        model: AbstractModel,
    ) -> TermTree[AbstractLoss]:
        """Evaluate, weight, and return all component terms.

        Arguments:
            states: Trajectories of system states for a set of trials.
            trial_specs: Task specifications for the set of trials.
        """
        children = {}
        for name, loss in self.terms.items():
            node = loss(states, trial_specs, model)
            w = self.weights.get(name, 1.0)
            children[name] = node.with_weight(w)
        return TermTree.branch(self.label, children, originator=self)

    def without(self, *keys: str, label: Optional[str] = None) -> "CompositeLoss":
        """Return a new `CompositeLoss` without the specified terms."""
        return CompositeLoss(
            terms={k: v for k, v in self.terms.items() if k not in keys},
            weights={k: v for k, v in self.weights.items() if k not in keys},
            label=self.label if label is None else label,
        )

    # def skeleton(self, batch_dims: tuple[int, ...]) -> TermTree:
    #     children: dict[str, TermTree] = {}
    #     for name, loss in self.terms.items():
    #         child = loss.skeleton(batch_dims)
    #         child = child.with_weight(self.weights.get(name, 1.0))
    #         children[name] = child
    #     return TermTree.branch(self.label, children, originator=self)


# Maybe rename TargetValueSpec; I feel like a "`TargetSpec`" would include a `where` field
class TargetSpec(Module):
    """Associate a state's target value with time indices and discounting factors."""

    # `value` may be `None` when we specify default values for the other fields
    value: Optional[PyTree[Array]] = None
    time_idxs: Optional[Array | Callable] = None
    time_mask: Optional[Array | Callable] = None
    discount: Optional[Array | Callable] = None  # field(default_factory=lambda: jnp.array([1.0]))

    def __and__(self, other):
        # Allows user to do `target_zero & target_final_state`, for example.
        return eqx.combine(self, other)

    def __rand__(self, other):
        # Necessary for edge case of `None & spec`
        return eqx.combine(other, self)

    def get_time_mask(self, n: int) -> Optional[Array | Callable]:
        if self.time_idxs is None:
            return None
        mask = jnp.zeros((n,), dtype=bool)

        if callable(self.time_idxs):

            def mask_fn(trial_spec):
                idxs = self.time_idxs(trial_spec)  # type: ignore
                return mask.at[idxs].set(True)

            return mask_fn

        else:
            mask = jnp.zeros((n,), dtype=bool)
            mask = mask.at[self.time_idxs].set(True)
            return mask

    @property
    def batch_axes(self) -> PyTree[None | int]:
        # Assume that only the target value will vary between trials.
        # TODO: (Low priority.) It's probably better to give control over this to
        # `AbstractTask`, since in some cases we might want to vary these parameters
        # over trials and not just across batches. And if we don't want to vary them
        # at all, then why are time_mask and discount not just fields of
        # `TargetStateLoss`?
        return TargetSpec(
            value=0,
            time_mask=None,
            discount=None,
        )


"""Useful partial target specs"""
target_final_state = TargetSpec(None, jnp.array([-1], dtype=int), None)
target_zero = TargetSpec(jnp.array(0.0), None, None)


class TargetStateLoss(AbstractLoss):
    """Penalize a state variable in comparison to a target value.

    !!! Note ""
        Currently only supports `where` functions that select a
        single state array, not a `PyTree[Array]`.

    Arguments:
        label: The label for the loss term.
        where: Function that takes the PyTree of model states, and
            returns the substate to be penalized.
        norm: Function which takes the difference between
            the substate and the target, and transforms it into a distance. For example,
            if the substate is effector position, then the substate-target difference
            gives the difference between the $x$ and $y$ position components, and the
            default `norm` function (`jnp.linalg.norm` on `axis=-1`) returns the
            Euclidean distance between the actual and target positions.
        spec: Gives default/constant values for the substate target, discount, and
            time index.
    """

    label: str
    where: Callable
    norm: Callable = lambda x: jnp.sum(x**2, axis=-1)
    # norm: Callable = lambda x: jnp.linalg.norm(x, axis=-1)  # Spatial distance
    spec: Optional[TargetSpec] = None  # Default/constant values.

    @cached_property
    def key(self):
        return WhereDict.key_transform(self.where)

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        """
        Arguments:
            trial_specs: Trial-by-trial information. In particular, if
                `trial_specs.targets` contains a `TargetSpec` entry mapped by
                `self.key`, the values of that `TargetSpec` instance will
                take precedence over the defaults specified by `self.spec`.
                This allows `AbstractTask` subclasses to specify trial-by-trial
                targets, where appropriate.
        """
        assert states is not None, "TargetStateLoss requires states, but states is None"
        assert trial_specs is not None, (
            "TargetStateLoss requires trial_specs, but trial_specs is None"
        )

        # TODO: Support PyTrees, not just single arrays
        state = self.where(states)[:, 1:]

        if (task_target_spec := trial_specs.targets.get(self.key, None)) is None:
            if self.spec is None:
                raise ValueError(
                    "`TargetSpec` must be provided on construction of "
                    "`TargetStateLoss`, or as part of the trial "
                    "specifications"
                )

            target_spec = self.spec
        elif isinstance(task_target_spec, TargetSpec):
            # Override default spec with trial-by-trial spec provided by the task, if any
            target_spec: TargetSpec = eqx.combine(self.spec, task_target_spec)
        elif isinstance(task_target_spec, Mapping):
            target_spec: TargetSpec = eqx.combine(self.spec, task_target_spec[self.label])
        else:
            raise ValueError("Invalid target spec encountered ")

        loss_over_time = self.norm(state - target_spec.value)

        # https://chatgpt.com/share/68ec227e-052c-8006-86ac-ffc5dc490b4d

        time_mask = target_spec.time_mask
        if time_mask is None:
            time_mask = target_spec.get_time_mask(loss_over_time.shape[-1])

        masks = [x for x in [time_mask, target_spec.discount] if x is not None]

        # ? Should we keep the weights?
        return reduce_over_time_with_weights(
            label=self.label,
            arr=loss_over_time,
            trial_specs=trial_specs,
            time_axis=-1,
            trial_axis=0,  #! Correct?
            trial_axis_specs=0,  # in `trial_specs`
            masks=masks,
        )


WeightsSpec = float | Array | Callable[[object], float | Array]
# Callable takes a single trial's spec (PyTree leaf view) and returns scalar or (T,) weights


def _move_trial_axis_pytree(tree, trial_axis: int):
    def _move(x):
        if not isinstance(x, Array) or x.ndim < 2:
            return x
        return jnp.moveaxis(x, trial_axis, 0)

    return jt.map(_move, tree)


def _per_trial_weights(selector: WeightsSpec, specs_T0, T: int, dtype) -> Array:
    """Return (N, T) float weights from scalar/(T,) array OR callable(spec_i)->scalar/(T,)."""
    leaves = jt.leaves(specs_T0)
    N = leaves[0].shape[0] if leaves else 1

    def _as_T(wi):
        wi = jnp.asarray(wi)
        if wi.ndim == 0:
            return jnp.full((T,), wi.astype(dtype), dtype=dtype)
        if wi.shape == (T,):
            return wi.astype(dtype)
        raise ValueError(f"weight must be scalar or shape (T,), got {wi.shape}")

    if not callable(selector):
        w = _as_T(selector)
        return jnp.broadcast_to(w, (N, T))

    def one(spec_i):
        return _as_T(selector(spec_i))

    return eqx.filter_vmap(one)(specs_T0)  # (N, T)


def _combine_weights(
    selectors: Sequence[WeightsSpec],
    specs_T0,
    T: int,
    dtype,
) -> Array:
    """Multiply an arbitrary number of weight specs into a single (N,T) weight array."""
    if not selectors:
        N = jtu.tree_leaves(specs_T0)[0].shape[0] if jtu.tree_leaves(specs_T0) else 1
        return jnp.ones((N, T), dtype=dtype)
    w = _per_trial_weights(selectors[0], specs_T0, T, dtype)
    for s in selectors[1:]:
        w = w * _per_trial_weights(s, specs_T0, T, dtype)
    return w


def _trial_time_perm(ndim: int, trial_axis: int, time_axis: int):
    """
    Permutation that moves TRIAL -> 0 and TIME -> -1 while preserving the
    relative order of all other axes.
    """
    ta = trial_axis if trial_axis >= 0 else ndim + trial_axis
    ti = time_axis if time_axis >= 0 else ndim + time_axis
    others = [ax for ax in range(ndim) if ax not in (ta, ti)]
    # Preserve original order of the non-(trial,time) axes
    return [ta, *others, ti]


def _inv_perm(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def reduce_over_time_with_weights(
    *,
    label: str,
    arr: Array,  # shape: (..., T, F?) pre-norm
    trial_specs,  # PyTree; each leaf has a trial axis
    time_axis: int,  # where T lives in `arr`
    trial_axis: int,  # where trial axis lives in `arr`
    trial_axis_specs: int,  # where trial axis lives in `trial_specs` leaves
    masks: Sequence[WeightsSpec],  # any number of scalar/(T,) or callable(spec_i)->scalar/(T,)
) -> Array:
    """
    1) Transpose so: (trial, ..., time).
    2) Build combined per-trial weights W (N, T) by multiplying all masks.
    3) Weighted sum over time, restore trial axis, drop time axis.
    """
    # Put trial at 0 and time at -1 in one explicit transpose
    perm = _trial_time_perm(arr.ndim, trial_axis, time_axis)
    inv_perm = _inv_perm(perm)
    arr_rt = jnp.transpose(arr, perm)  # (N, ..., T)
    specs_T0 = _move_trial_axis_pytree(trial_specs, trial_axis_specs)

    T = arr_rt.shape[-1]
    dtype = arr_rt.dtype

    # Flatten non-time dims (except N) so (N, T) weights broadcast cleanly
    N = arr_rt.shape[0]
    rest = arr_rt.shape[1:-1]
    arr_flat = arr_rt.reshape((N, -1, T))  # (N, R, T)

    # Combine all masks/discounts/etc. as multiplicative weights
    W = _combine_weights(masks, specs_T0, T, dtype)  # (N, T)

    # Print the indices of the first zero entry in each row (trial) of the weights
    # jax.debug.print("{a}\n{b}\n\n", a=label, b=jnp.argmax(W == 0, axis=1))

    # Weighted reduction over time
    reduced_flat = (arr_flat * W[:, None, :]).sum(axis=-1)  # (N, R)
    reduced = reduced_flat.reshape((N, *rest))  # (N, ...)

    # Restore the trial axis to its original slot in the *output* (time removed)
    # inv_perm maps (trial, others..., time) -> original. Drop the 'time' entry,
    # then move axis 0 to where 'trial' originally lived.
    inv_perm_wo_time = [p for i, p in enumerate(inv_perm) if i != (len(inv_perm) - 1)]
    target_trial_axis = inv_perm_wo_time.index(0)
    out = jnp.moveaxis(reduced, 0, target_trial_axis)
    return out


"""Penalizes the effector's squared distance from the target position
across the trial."""
effector_pos_loss = TargetStateLoss(
    "Effector position",
    where=lambda state: state.mechanics.effector.pos,
    # Euclidean distance
    norm=lambda *args, **kwargs: (jnp.linalg.norm(*args, axis=-1, **kwargs) ** 2),
)


effector_vel_loss = TargetStateLoss(
    "Effector position",
    where=lambda state: state.mechanics.effector.vel,
    # Euclidean distance
    norm=lambda *args, **kwargs: (jnp.linalg.norm(*args, axis=-1, **kwargs) ** 2),
    spec=target_final_state,
)


class StopAtGoalLoss(AbstractLoss):
    """Encourages the effector to stop at the goal at least once.

    This is different from the typical "be at goal by the end of trial" loss.

    Two methods are provided: "softmin" and "soft-or". Soft-or is based only on the existence of
    a time window where the effector is at the goal, and so may be preferred if we do not want time
    pressure (aside from what is implied by `window_len` versus episode length). Softmin may or may
    not add some time pressure, depending on the other parameters.
    """

    label: str = "effector_stop_at_goal"
    method: Literal["softmin", "soft-or"] = "softmin"
    std_dist: float = 0.025  # how close to goal is "at goal"
    std_vel: float = 0.05
    vel_weight: float = 1.0
    window_len: int = 5  # min no. steps required to be "at goal"
    # softmin: how soft the min is (small -> credit focused on a single window)
    # soft-or: how close to the goal we need to be to register as a success
    tau: float = 0.2
    eps: float = 1e-6  # to avoid log(0)

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        assert states is not None, "StopAtGoalLoss requires states"
        assert trial_specs is not None, "StopAtGoalLoss requires trial_specs"
        # assert trial_specs is not None, "StopAtGoalLoss requires trial_specs"

        pos = states.mechanics.effector.pos[:, 1:]
        vel = states.mechanics.effector.vel[:, 1:]
        goal = trial_specs.targets["mechanics.effector.pos"].value  # [B, 2]

        dist_sq = jnp.sum((pos - goal) ** 2, axis=-1)
        vel_sq = jnp.sum(vel**2, axis=-1)
        c = dist_sq / (self.std_dist**2) + self.vel_weight * vel_sq / (self.std_vel**2)  # [T+1]
        c_avgs = moving_avg(c, self.window_len)  # [T+1-K]

        if self.method == "softmin":
            return softmin(c_avgs, self.tau)  # scalar
        elif self.method == "soft-or":
            s = jnp.clip(jax.nn.sigmoid(-c_avgs / self.tau), 1e-12, 1.0 - 1e-12)
            log1m_s = jnp.log1p(-(s - self.eps))
            log_prod = jnp.sum(log1m_s)
            p_exist = 1.0 - jnp.exp(log_prod)
            return -jnp.log(p_exist + self.eps)
        else:
            raise ValueError("method must be 'softmin' or 'soft-or'")


class ModelLoss(AbstractLoss):
    """Wrapper for functions that take a model, and return a scalar."""

    label: str
    loss_fn: Callable[[AbstractModel], Array]

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        assert model is not None, "ModelLoss requires a model, but model is None"
        return self.loss_fn(model)


class EffectorStraightPathLoss(AbstractLoss):
    """Penalizes non-straight paths followed by the effector between initial
    and final position.

    !!! Info ""
        Calculates the length of the paths followed, and normalizes by the
        Euclidean (straight-line) distance between the initial and final state.

    Attributes:
        label: The label for the loss term.
        normalize_by: Controls whether to normalize by the distance between the
            initial position & actual final position, or the initial position
            & task-specified goal position.
    """

    label: str = "effector_path_straightness"
    normalize_by: Literal["actual", "goal"] = "actual"

    def term(
        self,
        states: Optional["SimpleFeedbackState"],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        assert states is not None, "EffectorStraightPathLoss requires states"
        assert trial_specs is not None, "EffectorStraightPathLoss requires trial_specs"

        effector_pos = states.mechanics.effector.pos
        pos_diff = jnp.diff(effector_pos, axis=1)
        piecewise_lengths = jnp.linalg.norm(pos_diff, axis=-1)
        path_length = jnp.sum(piecewise_lengths, axis=1)
        if self.normalize_by == "actual":
            final_pos = effector_pos[:, -1]
        elif self.normalize_by == "goal":
            final_pos = trial_specs.targets["mechanics.effector"].value.pos
        else:
            raise ValueError("normalize_by must be 'actual' or 'goal'")
        init_final_diff = final_pos - effector_pos[:, 0]
        straight_length = jnp.linalg.norm(init_final_diff, axis=-1)

        loss = path_length / straight_length

        return loss


def power_discount(n_steps: int, discount_exp: int = 6) -> Array:
    """A power-law vector that puts most of the weight on its later elements.

    Arguments:
        n_steps: The number of time steps in the trajectory to be weighted.
        discount_exp: The exponent of the power law.
    """
    if discount_exp == 0:
        return jnp.array(1.0)
    else:
        return jnp.linspace(1.0 / n_steps, 1.0, n_steps) ** discount_exp
