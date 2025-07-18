"""Composable loss function modules operating on state PyTrees.

TODO:

- `LossDict` only computes the total loss once, but when we append a `LossDict`
  for a single timestep to `losses_history` in `TaskTrainer`, we lose the loss
  total for that time step. When it is needed later (e.g. on plotting the loss)
  it will be recomputed, once. It is also not serialized along with the
  `losses_history`. I doubt this is a significant computational loss
  (how many loss terms * training iterations could be involved? 1e6?)
  to have to compute from time to time, but perhaps it would be nice to
  include the total as part of flatten/unflatten. It'd probably just require
  that we allow passing the total on instantiation, however that would be kind
  of weird.
    - Even if we have 6 loss terms with 1e6 iterations, it only takes ~130 ms
    to compute `losses.total`. Given that we only need to compute this once
    per session or so, it shouldn't be a problem.
- The time aggregation could be done in `CompositeLoss`, if we unsqueeze
  terms that don't have a time dimension. This would allow time aggregation
  to be controlled in one place, if for some reason it makes sense to change
  how this aggregation occurs across all loss terms.
- Protocols for all the different `state` types/fields?
    - Alternatively we could make `AbstractLoss` generic over a
      `StateT` typevar, however that might not make sense for typing
      the compositions (e.g. `__sum__`) since the composite can support any
      state pytrees that have the right combination of fields, not just pytrees
      that have an identical structure.
- L2 by default, but should allow for other norms

:copyright: Copyright 2023-2024 by Matt Laporte.
:license: Apache 2.0. See LICENSE for details.
"""

#! Can't do this because `AbstractVar` annotations can't be stringified.
# from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property, partial
import logging
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Tuple,
)

import equinox as eqx
from equinox import AbstractVar, Module, field
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float, PyTree

from feedbax._model import AbstractModel
from feedbax.misc import get_unique_label, unzip2
from feedbax._mapping import WhereDict
from feedbax.state import State

if TYPE_CHECKING:
    from feedbax.bodies import SimpleFeedbackState
    from feedbax.task import TaskTrialSpec


logger = logging.getLogger(__name__)


@jtu.register_pytree_node_class
class LossDict(dict[str, Array]):
    """Dictionary that provides a sum over its values."""

    @cached_property
    def total(self) -> Array:
        """Elementwise sum over all values in the dictionary."""
        loss_term_values = list(self.values())
        return jax.tree_util.tree_reduce(lambda x, y: x + y, loss_term_values)
        # return jnp.sum(jtu.tree_map(
        #     lambda *args: sum(args),
        #     *loss_term_values
        # ))

    def __setitem__(self, key, value):
        raise TypeError("LossDict does not support item assignment")

    def update(self, other=(), /, **kwargs):
        raise TypeError("LossDict does not support update")

    def tree_flatten(self):
        """The same flatten function used by JAX for `dict`"""
        return unzip2(sorted(self.items()))[::-1]

    @classmethod
    def tree_unflatten(cls, keys, values):
        return LossDict(zip(keys, values))


def is_lossdict(x):
    return isinstance(x, LossDict)


class AbstractLoss(Module):
    """Abstract base class for loss functions.

    Instances can be composed by addition and scalar multiplication.
    """

    label: AbstractVar[str]

    def __call__(
        self,
        states: PyTree,
        trial_specs: "TaskTrialSpec",
        model: AbstractModel,
    ) -> LossDict:
        return LossDict({self.label: self.term(states, trial_specs, model)})

    @abstractmethod
    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        """Implement this to calculate a loss term."""
        ...

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


class CompositeLoss(AbstractLoss):
    """Incorporates multiple simple loss terms and their relative weights."""

    terms: dict[str, AbstractLoss]
    weights: dict[str, float]
    label: str

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
            raise ValueError(
                "Mismatch between number of loss terms and number of term weights"
            )

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
        term_tuples_leaves = jax.tree_map(
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
    ) -> LossDict:
        """Evaluate, weight, and return all component terms.

        Arguments:
            states: Trajectories of system states for a set of trials.
            trial_specs: Task specifications for the set of trials.
        """
        # Evaluate all loss terms
        losses = jax.tree_map(
            lambda loss: loss.term(states, trial_specs, model),
            self.terms,
            is_leaf=lambda x: isinstance(x, AbstractLoss),
        )

        # aggregate over batch for state-based losses
        losses = jax.tree_map(
            lambda x: jnp.mean(x, axis=0) if x.ndim > 0 else x,
            losses,
        )

        if self.weights is not None:
            # term scaling
            losses = jax.tree_map(
                lambda term, weight: term * weight,
                dict(losses),
                self.weights,
            )

        return LossDict(losses)

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        return self(states, trial_specs, model).total

# Maybe rename TargetValueSpec; I feel like a "`TargetSpec`" would include a `where` field
class TargetSpec(Module):
    """Associate a state's target value with time indices and discounting factors."""
    # `value` may be `None` when we specify default values for the other fields
    value: Optional[PyTree[Array]] = None
    # TODO: If `time_idxs` is `Array`, it must be 1D or we'll lose the time dimension before we sum over it!
    time_idxs: Optional[Array] = None
    discount: Optional[Array] = None # field(default_factory=lambda: jnp.array([1.0]))

    def __and__(self, other):
        # Allows user to do `target_zero & target_final_state`, for example.
        return eqx.combine(self, other)

    def __rand__(self, other):
        # Necessary for edge case of `None & spec`
        return eqx.combine(other, self)

    @property
    def batch_axes(self) -> PyTree[None | int]:
        # Assume that only the target value will vary between trials.
        # TODO: (Low priority.) It's probably better to give control over this to
        # `AbstractTask`, since in some cases we might want to vary these parameters
        # over trials and not just across batches. And if we don't want to vary them
        # at all, then why are time_idxs and discount not just fields of
        # `TargetStateLoss`?
        return TargetSpec(
            value=0,
            time_idxs=None,
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
        assert trial_specs is not None, "TargetStateLoss requires trial_specs, but trial_specs is None"

        # TODO: Support PyTrees, not just single arrays
        state = self.where(states)[:, 1:]

        if (task_target_spec := trial_specs.targets.get(self.key, None)) is None:
            if self.spec is None:
                raise ValueError("`TargetSpec` must be provided on construction of "
                                 "`TargetStateLoss`, or as part of the trial "
                                 "specifications")

            target_spec = self.spec
        elif isinstance(task_target_spec, TargetSpec):
            # Override default spec with trial-by-trial spec provided by the task, if any
            target_spec: TargetSpec = eqx.combine(self.spec, task_target_spec)
        elif isinstance(task_target_spec, Mapping):
            target_spec: TargetSpec = eqx.combine(self.spec, task_target_spec[self.label])
        else:
            raise ValueError("Invalid target spec encountered ")

        if target_spec.time_idxs is not None:
            state = state[..., target_spec.time_idxs, :]

        loss_over_time = self.norm(state - target_spec.value)

        # jax.debug.print("loss_over_time\n{a}\n\nstate\n{b}\n\n\n\n\n", a=loss_over_time, b=state)

        if target_spec.discount is not None:
            loss_over_time = loss_over_time * target_spec.discount

        return jnp.sum(loss_over_time, axis=-1)


"""Penalizes the effector's squared distance from the target position
across the trial."""
effector_pos_loss = TargetStateLoss(
    "Effector position",
    where=lambda state: state.mechanics.effector.pos,
    # Euclidean distance
    norm=lambda *args, **kwargs: (
        jnp.linalg.norm(*args, axis=-1, **kwargs) ** 2
    ),
)


effector_vel_loss = TargetStateLoss(
    "Effector position",
    where=lambda state: state.mechanics.effector.vel,
    # Euclidean distance
    norm=lambda *args, **kwargs: (
        jnp.linalg.norm(*args, axis=-1, **kwargs) ** 2
    ),
    spec=target_final_state,
)


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


class EffectorPositionLoss(AbstractLoss):
    label: str = "Effector position"
    discount_func: Callable[[int], Float[Array, "#time"]] = (
        lambda n_steps: power_discount(n_steps, discount_exp=6)[None, :]
    )

    def term(
        self,
        states: Optional["SimpleFeedbackState"],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        assert states is not None, "EffectorPositionLoss requires states"
        assert trial_specs is not None, "EffectorPositionLoss requires trial_specs"

        # Sum over X, Y, giving the squared Euclidean distance
        loss = jnp.sum(
            (states.mechanics.effector.pos[:, 1:] - trial_specs.target.pos) ** 2, axis=-1  # type: ignore
        )

        # temporal discount
        if self.discount_func is not None:
            loss = loss * self.discount(loss.shape[-1])

        # sum over time
        loss = jnp.sum(loss, axis=-1)

        return loss

    def discount(self, n_steps):
        # Can't use a cache because of JIT.
        # But we only need to run this once per training iteration...
        return self.discount_func(n_steps)


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

    label: str = "Effector path straightness"
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


class EffectorFixationLoss(AbstractLoss):
    """Penalizes the effector's squared distance from the fixation position.

    !!! Info ""
        Similar to `EffectorPositionLoss`, but only penalizes the position
        error during the part of the trial where `trial_specs.inputs.hold`
        is non-zero/`True`.

    Attributes:
        label: The label for the loss term.
    """

    label: str = "Effector maintains fixation"

    def term(
        self,
        states: Optional[PyTree],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        assert states is not None, "EffectorFixationLoss requires states"
        assert trial_specs is not None, "EffectorFixationLoss requires trial_specs"

        loss = jnp.sum(
            (states.mechanics.effector.pos[:, 1:] - trial_specs.target.pos) ** 2, axis=-1
        )

        loss = loss * jnp.squeeze(trial_specs.inputs.hold)  # type: ignore

        # sum over time
        loss = jnp.sum(loss, axis=-1)

        return loss


class EffectorVelocityLoss(AbstractLoss):
    """Penalizes the squared difference in effector velocity relative to the target
    velocity across the trial.

    Attributes:
        label: The label for the loss term.
        discount_func: Returns a trajectory with which to weight (discount)
            the loss values calculated for each time step of the trial.
            Defaults to a power-law curve that puts most of the weight on
            time steps near the end of the trial.
    """

    label: str = "Effector position"
    discount_func: Callable[[int], Float[Array, "#time"]] = lambda n_steps: jnp.float32(1.0)

    def term(
        self,
        states: Optional["SimpleFeedbackState"],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        assert states is not None, "EffectorVelocityLoss requires states"
        assert trial_specs is not None, "EffectorVelocityLoss requires trial_specs"

        # Sum over X, Y, giving the squared Euclidean distance
        loss = jnp.sum(
            (states.mechanics.effector.vel[:, 1:] - trial_specs.target.vel) ** 2, axis=-1  # type: ignore
        )

        # temporal discount
        if self.discount_func is not None:
            loss = loss * self.discount(loss.shape[-1])

        # sum over time
        loss = jnp.sum(loss, axis=-1)

        return loss

    def discount(self, n_steps):
        # Can't use a cache because of JIT.
        # But we only need to run this once per training iteration...
        return self.discount_func(n_steps)


class EffectorFinalVelocityLoss(AbstractLoss):
    """Penalizes the squared difference between the effector's final velocity
    and the goal velocity (typically zero) on the final timestep.

    Attributes:
        label: The label for the loss term.
    """

    label: str = "Effector final velocity"

    def term(
        self,
        states: Optional["SimpleFeedbackState"],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        assert states is not None, "EffectorFinalVelocityLoss requires states"
        assert trial_specs is not None, "EffectorFinalVelocityLoss requires trial_specs"

        loss = jnp.sum(
            (states.mechanics.effector.vel[:, -1] - trial_specs.target.vel[:, -1]) ** 2,
            axis=-1,
        )

        return loss


class NetworkOutputLoss(AbstractLoss):
    """Penalizes the squared values of the network's outputs.

    Attributes:
        label: The label for the loss term.
    """

    label: str = "NN output"

    def term(
        self,
        states: Optional["SimpleFeedbackState"],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        assert states is not None, "NetworkOutputLoss requires states"
        assert trial_specs is not None, "NetworkOutputLoss requires trial_specs"

        assert states.net.output is not None, (
            "Cannot calculate NetworkOutputLoss if network output is None"
        )

        # Sum over output channels
        loss = jnp.sum(states.net.output**2, axis=-1)

        # Sum over time
        loss = jnp.sum(loss, axis=-1)

        return loss


class NetworkActivityLoss(AbstractLoss):
    """Penalizes the squared values of the network's hidden activity.

    Attributes:
        label: The label for the loss term.
    """

    label: str = "NN hidden activity"

    def term(
        self,
        states: Optional["SimpleFeedbackState"],
        trial_specs: Optional["TaskTrialSpec"],
        model: Optional[AbstractModel],
    ) -> Array:
        assert states is not None, "NetworkActivityLoss requires states"
        assert trial_specs is not None, "NetworkActivityLoss requires trial_specs"

        # Sum over hidden units
        loss = jnp.sum(states.net.hidden**2, axis=-1)

        # sum over time
        loss = jnp.sum(loss, axis=-1)

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


def mse(x, y):
    """Mean squared error."""
    return jax.tree_map(
        lambda x, y: jnp.mean((x - y) ** 2),
        x,
        y,
    )


def nan_safe_mse(
    preds: Array,
    targets: Array,
) -> Array:
    """
    Calculates MSE safely for gradients when targets have NaN entries.

    Assumes that if `pred` has NaN entries, then `target` will also have NaNs in the same rows.

    Computes a mask of the NaN entries in `targets`, replaces NaNs with zeros,
    proceeds with MSE calculation, then masks the NaN entries out of the result
    prior to aggregation.
    """
    valid_mask = ~jnp.isnan(targets)
    targets_cleaned = jnp.nan_to_num(targets, nan=0.0)
    squared_errors = (preds - targets_cleaned)**2
    masked_squared_errors = jnp.where(valid_mask, squared_errors, 0.0)
    sum_of_squared_errors = jnp.sum(masked_squared_errors)
    num_valid_elements = jnp.sum(valid_mask)
    return sum_of_squared_errors / jnp.maximum(num_valid_elements, 1.0)
