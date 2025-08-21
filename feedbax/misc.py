"""Tools which did not belong any particular other place.

:copyright: Copyright 2023-2024 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Set,
)
import copy
import difflib
import dis
from functools import wraps
import inspect
from itertools import zip_longest, chain
import logging
from operator import attrgetter
import os
from pathlib import Path, PosixPath
from shutil import rmtree
import subprocess
import textwrap
from time import perf_counter
from types import ModuleType
from typing import Any, Optional, Tuple, TypeVar, Union

import equinox as eqx
from equinox import Module
# from equinox._pretty_print import tree_pp, bracketed
import jax
import jax.numpy as jnp
import jax.tree as jt
import jax._src.pretty_printer as pp
import jax.tree_util as jtu
import jax.tree as jt
from jaxtyping import Float, Array, PyTree, Shaped

from feedbax._progress import _tqdm_write


logger = logging.getLogger(__name__)


"""The signs of the i-th derivatives of cos and sin.

TODO: infinite cycle
"""
SINCOS_GRAD_SIGNS = jnp.array([(1, 1), (1, -1), (-1, -1), (-1, 1)])


T1 = TypeVar("T1")
T2 = TypeVar("T2")


class Timer:
    """Context manager for timing code blocks.

    Derived from https://stackoverflow.com/a/69156219
    """

    def __init__(self):
        self.times = []

    def __enter__(self, printout=False):
        self.start_time = perf_counter()
        self.printout = printout
        return self  # Apparently you're supposed to only yield the key

    def __exit__(self, *args, **kwargs):
        self.time = perf_counter() - self.start_time
        self.times.append(self.time)
        self.readout = f"Time: {self.time:.3f} seconds"
        if self.printout:
            print(self.readout)

    start = __enter__
    stop = __exit__


class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console.

    Source: https://stackoverflow.com/a/67257516
    """

    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            _tqdm_write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


class StrAlwaysLT(str):

    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False

    # def __repr__(self):
    #     return self.replace("'", "")


class BatchInfo(eqx.Module):
    size: int
    current: int
    total: int
    start: int = 0

    @property
    def progress(self) -> float:
        return self.current / self.total

    @property
    def run_progress(self) -> float:
        return (self.current - self.start) / (self.total - self.start)


def delete_contents(path: Union[str, Path]):
    """Delete all subdirectories and files of `path`."""
    for p in Path(path).iterdir():
        if p.is_dir():
            rmtree(p)
        elif p.is_file():
            p.unlink()


def _dirname_of_this_module():
    """Return the directory containing this module."""
    return os.path.dirname(os.path.abspath(__file__))


def git_commit_id(
    path: Optional[str | PosixPath] = None,
    module: Optional[ModuleType] = None,
) -> str:
    """Get the ID of the currently checked-out commit in the repo at `path`.

    If no `path` or `module` is given, returns the commit ID for the repo
    containing this function definition.

    Derived from <https://stackoverflow.com/a/57683700>
    """
    if path is None:
        if module is None:
            path = _dirname_of_this_module()
        else:
            path = Path(module.__file__).absolute().parent

    commit_id = (
        subprocess.check_output(["git", "describe", "--always"], cwd=path)
        .strip()
        .decode()
    )

    return commit_id


def identity_func(x):
    """The identity function."""
    return x


def n_positional_args(func: Callable) -> int:
    """Get the number of positional arguments of a function."""
    sig = inspect.signature(func)
    return sum(
        1
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    )


def interleave_unequal(*args):
    """Interleave sequences of different lengths."""
    return (x for x in chain.from_iterable(zip_longest(*args)) if x is not None)


def corners_2d(bounds: Float[Array, "2 xy=2"]):
    """Generate the corners of a rectangle from its bounds."""
    xy = jt.map(jnp.ravel, jnp.meshgrid(*bounds.T))
    return jnp.vstack(xy)


def indent_str(s: str, indent: int = 4) -> str:
    """Pretty format a PyTree, but indent all lines with `indent` spaces."""
    indent_str = " " * indent
    return indent_str + s.replace("\n", f"\n{indent_str}")


def unzip2(xys: Iterable[Tuple[T1, T2]]) -> Tuple[Tuple[T1, ...], Tuple[T2, ...]]:
    """Unzip sequence of length-2 tuples into two tuples.

    Taken from `jax._src.util`.
    """
    # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
    # is too permissive about inputs, and does not guarantee a length-2 output.
    xs: MutableSequence[T1] = []
    ys: MutableSequence[T2] = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return tuple(xs), tuple(ys)


def get_unique_label(label: str, invalid_labels: Sequence[str] | Set[str]) -> str:
    """Get a unique string from a base string, while avoiding certain strings.

    Simply appends consecutive integers to the string until a unique string is
    found.
    """
    i = 0
    label_ = label
    while label_ in invalid_labels:
        label_ = f"{label}_{i}"
        i += 1
    return label_


def highlight_string_diff(obj1, obj2):
    """Given two objects, give a string that highlights the differences in
    their string representations.

    This can be useful for identifying slight differences in large PyTrees.

    Source: https://stackoverflow.com/a/76946768
    """
    str1 = repr(obj1)
    str2 = repr(obj2)

    matcher = difflib.SequenceMatcher(None, str1, str2)

    str2_new = ""
    i = 0
    for m in matcher.get_matching_blocks():
        if m.b > i:
            str2_new += str2[i : m.b]
        str2_new += f"\033[91m{str2[m.b:m.b + m.size]}\033[0m"
        i = m.b + m.size

    return str2_new.replace('\\n', '\n')


def print_trees_side_by_side(tree1, tree2, column_width=60, separator='|'):
    """Given two PyTrees, print their pretty representations side-by-side."""
    strs1 = eqx.tree_pformat(tree1).split('\n')
    strs2 = eqx.tree_pformat(tree2).split('\n')

    def wrap_text(text, width):
        return textwrap.wrap(text, width) or ['']

    wrapped1 = [wrap_text(s, column_width) for s in strs1]
    wrapped2 = [wrap_text(s, column_width) for s in strs2]

    for w1, w2 in zip_longest(wrapped1, wrapped2, fillvalue=['']):
        max_lines = max(len(w1), len(w2))

        for i in range(max_lines):
            line1 = w1[i] if i < len(w1) else ''
            line2 = w2[i] if i < len(w2) else ''
            print(f"{line1:<{column_width}} {separator} {line2:<{column_width}}")


def unique_generator(
    seq: Sequence[T1],
    replace_duplicates: bool = False,
    replace_value: Any = None
) -> Iterable[Optional[T1]]:
    """Yields the first occurrence of sequence entries, in order.

    If `replace_duplicates` is `True`, replaces duplicates with `replace_value`.
    """
    seen = set()
    for item in seq:
        if id(item) not in seen:
            seen.add(id(item))
            yield item
        elif replace_duplicates:
            yield replace_value


def is_module(element: Any) -> bool:
    """Return `True` if `element` is an Equinox module."""
    return isinstance(element, Module)


def is_none(x):
    return x is None


def nested_dict_update(dict_, *args, make_copy: bool = True):
    """Source: https://stackoverflow.com/a/3233356/23918276"""
    if make_copy:
        dict_ = copy.deepcopy(dict_)
    for arg in args:
        for k, v in arg.items():
            if isinstance(v, Mapping):
                dict_[k] = nested_dict_update(
                    dict_.get(k, type(v)()),
                    v,
                    make_copy=make_copy,
                )
            else:
                dict_[k] = v
    return dict_


# def _simple_module_pprint(name, *children, **kwargs):
#     return bracketed(
#         pp.text(name),
#         kwargs['indent'],
#         [tree_pp(child, **kwargs) for child in children],
#         '(',
#         ')'
#     )


def _get_where_str(where_func: Callable) -> str:
    """
    Returns a string representation of the (nested) attributes accessed by a function.

    Only works for functions that take a single argument, and return the argument
    or a single (nested) attribute accessed from the argument.
    """
    bytecode = dis.Bytecode(where_func)
    return ".".join(instr.argrepr for instr in bytecode if instr.opname == "LOAD_ATTR")


class _NodeWrapper:
    def __init__(self, value):
        self.value = value


class NodePath:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        return iter(self.path)


def where_func_to_paths(where: Callable, tree: PyTree):
    """
    Similar to `_get_where_str`, but:

    - returns node paths, not strings;
    - works for `where` functions that return arbitrary PyTrees of nodes;
    - works for arbitrary node access (e.g. dict keys, sequence indices)
      and not just attribute access.

    Limitations:

    - requires a PyTree argument;
    - assumes the same object does not appear as multiple nodes in the tree;
    - if `where` specifies a node that is a subtree, it cannot also specify a node
      within that subtree.

    See [this issue](https://github.com/mlprt/feedbax/issues/14).
    """
    tree = eqx.tree_at(where, tree, replace_fn=lambda x: _NodeWrapper(x))
    id_tree = jtu.tree_map(id, tree, is_leaf=lambda x: isinstance(x, _NodeWrapper))
    node_ids = where(id_tree)

    paths_by_id = {node_id: path for path, node_id in jtu.tree_leaves_with_path(
        jtu.tree_map(
            lambda x: x if x in jt.leaves(node_ids) else None,
            id_tree,
        )
    )}

    paths = jtu.tree_map(lambda node_id: NodePath(paths_by_id[node_id]), node_ids)

    return paths


class _WhereStrConstructor:

    def __init__(self, label: str = ""):
        self.label = label

    def __getitem__(self, key: Any):
        if isinstance(key, str):
            key = f"'{key}'"
        elif isinstance(key, type):
            key = key.__name__
        return _WhereStrConstructor("".join([self.label, f"[{key}]"]))

    def __getattr__(self, name: str):
        sep = "." if self.label else ""
        return _WhereStrConstructor(sep.join([self.label, name]))


def _get_where_str_constructor_label(x: _WhereStrConstructor) -> str:
    return x.label


def where_func_to_attr_str_tree(where: Callable) -> PyTree[str]:
    """Also similar to `_get_where_str` and `where_func_to_paths`, but:

    - Avoids complicated logic of parsing bytecode, or traversing pytrees;
    - Works for `where` functions that return arbitrary PyTrees of node references;
    - Runs significantly (10+ times) faster than the other solutions.
    """

    try:
        return jt.map(_get_where_str_constructor_label, where(_WhereStrConstructor()))
    except TypeError:
        raise TypeError("`where` must return a PyTree of node references")


def attr_str_tree_to_where_func(tree: PyTree[str]) -> Callable:
    """Reverse transformation to `where_func_to_labels`.

    Takes a PyTree of strings describing attribute accesses, and returns a function
    that returns a PyTree of attributes.
    """
    getters = jt.map(lambda s: attrgetter(s), tree)

    def where_func(obj):
        return jt.map(lambda g: g(obj), getters)

    return where_func


def nan_bypass(
    func: Optional[Callable[..., Any]] = None,
    *,
    axis: int = 0,
    argnums: int | Sequence[int] | None = None,
    filler: float = 0.0,          
):
    """
    Decorator that temporarily fills slices along `axis` which contain *any*
    NaN in the positional arguments indexed by `argnums`, applies the wrapped
    function to the remaining data, and finally re-inserts NaNs.

    Args:
    axis: axis along which rows/segments are considered
    argnums: which positional arguments to check for NaNs.
        - int      → single argument
        - sequence → those indices
        - None     → all positional arguments
    filler: the value used to overwrite NaN rows before calling `func`
    """
    # allow decorator to be used with/without parentheses
    if func is not None and callable(func):
        return nan_bypass(axis=axis, argnums=argnums, filler=filler)(func)

    def decorator(f: Callable[..., Any]):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # 1. Resolve which arguments we have to inspect
            if argnums is None:
                argnums_tuple = tuple(range(len(args)))
            elif isinstance(argnums, int):
                argnums_tuple = (argnums,)
            else:
                argnums_tuple = tuple(argnums)

            # 2. Build a boolean mask of slices that contain *any* NaN
            def _row_has_nan(arr: jnp.ndarray) -> jnp.ndarray:
                red_axes = tuple(i for i in range(arr.ndim) if i != axis)
                return jnp.any(jnp.isnan(arr), axis=red_axes)

            nan_mask = jnp.zeros(args[argnums_tuple[0]].shape[axis], dtype=bool)
            for i in argnums_tuple:
                nan_mask = nan_mask | _row_has_nan(args[i])

            # 3. Broadcast mask to each target arg & replace with fillers
            def _replace_rows(arr: jnp.ndarray) -> jnp.ndarray:
                # make mask broadcastable to arr
                bmask = jnp.expand_dims(
                    nan_mask,
                    axis=tuple(ax for ax in range(arr.ndim) if ax != axis),
                )
                return jnp.where(bmask, jnp.zeros_like(arr) + filler, arr)

            safe_args = list(args)
            for i in argnums_tuple:
                safe_args[i] = _replace_rows(args[i])

            # 4. Call the original function
            out = f(*safe_args, **kwargs)

            # 5. Re-insert NaNs in the output on masked rows
            def _restore_rows(arr: jnp.ndarray) -> jnp.ndarray:
                bmask = jnp.expand_dims(
                    nan_mask,
                    axis=tuple(ax for ax in range(arr.ndim) if ax != axis),
                )
                nan_arr = jnp.full_like(arr, jnp.nan)
                return jnp.where(bmask, nan_arr, arr)

            return jt.map(_restore_rows, out)

        return wrapper

    return decorator


def batch_reshape(
    func: Optional[Callable[[Shaped[Array, "batch *n"]], Shaped[Array, "batch *m"]]] = None,
    *,
    n_nonbatch: int | Sequence[int] = 1,
):
    """Decorate a function to collapse its input array to a single batch dimension, and uncollapse the result.

    !!! Example
        Decorate `sklearn.decomposition.PCA.transform` so that it works on arrays with multiple
        batch dimensions.

        ```python
        # Generate some 30-dimension data points with two batch dims, and do 2-dim PCA
        n = 30
        data = np.random.random((10, 20, n))
        pca = PCA(n_components=2).fit(data.reshape(-1, data.shape[-1]))

        # Generate some more data we want to project, with an arbitrary number of batches
        more_data = np.random.rand((5, 10, 15, n))
        more_data_pc = batch_reshape(pca.transform)(more_data)  # (5, 10, 15, 2)
        ```

    !!! Note
        I originally added the `n_nonbatch` parameter in order to vmap a function over different arrays
        with different numbers of batch dimensions. Here's a trivial example.

        ```python
        key = jr.PRNGKeyArray(0)
        n = 5

        def func(arr: Shaped[Array, 'n']):
            # Do something; here we'll just add a trailing singleton dimensions
            return arr[..., None]

        arrays = (
            jr.normal(key, 10, 20, n),
            jr.normal(key, 50, 5, 16, n),
            jr.normal(key, 2, n),
        )

        # results is the same as arrays but each array has final singleton
        result = tree_map(batch_reshape(jax.vmap(func)), arrays)
        ```

        However, it turns out you can do this more easily with `jnp.vectorize`.

    Arguments:
        func: A function whose input and output arrays have a single batch dimension.
        n_nonbatch: The number of final axes that should not be collapsed and reformed. May be specified
            separately for each parameter.
    """
    def decorator(f):
        n_params = len(inspect.signature(f).parameters)

        if isinstance(n_nonbatch, int):
            n_nonbatch_tuple = (n_nonbatch,) * n_params
        elif isinstance(n_nonbatch, Sequence):
            assert len(n_nonbatch) == n_params, (
                "if n_nonbatch is a sequence it must have the same length "
                "as the number of parameters of func"
            )
            n_nonbatch_tuple = tuple(n_nonbatch)

        @wraps(f)
        def wrapper(*args):
            batch_shapes = set([arr.shape[:-n] for arr, n in zip(args, n_nonbatch_tuple)])
            assert len(batch_shapes) == 1, (
                "all input arrays must have the same batch shape"
            )
            batch_shape = batch_shapes.pop()

            # Reshape to collapse batch dimensions
            collapsed_args = tuple(
                arr.reshape((-1, *arr.shape[-n:])) for arr, n in zip(args, n_nonbatch_tuple)
            )

            result = f(*collapsed_args)

            # Reshape back to original batch structure
            return jt.map(
                lambda arr: arr.reshape((*batch_shape, *arr.shape[1:])),
                result,
            )

        return wrapper

    if func is None:
        # Called with parentheses: @batch_reshape() or @batch_reshape(n_nonbatch=2)
        return decorator
    else:
        # Called without parentheses: @batch_reshape
        return decorator(func)


def unkwargkey(f):
    """Converts a final `key` kwarg into an initial positional arg.

    This is useful because many Equinox modules take `key` as a kwarg, and Equinox
    transformations don't like kwargs -- but sometimes we want to transform over `key`
    anyway.
    """
    @wraps(f)
    def wrapper(key, *args):
        return f(*args, key=key)
    return wrapper


def batched_outer(
    x: Shaped[Array, "*batch n"],
    y: Shaped[Array, "*batch n"],
) -> Shaped[Array, "*batch n n"]:
    """Returns the outer product of the final dimension of an array."""
    return jnp.einsum('...i,...j->...ij', x, y)


def exponential_smoothing(
    x: Float[Array, "*"],
    alpha: float,
    init_window_size: int = 1,
    axis: int = -1,
):
    """
    Return the exponential moving average (EMA) of an array along the specified axis.

    Arguments:
        x: Input array
        alpha: Smoothing factor between 0 and 1.
        init_window_size: Initialize the EMA with the average of this many values
            from the beginning of the axis.
        axis: Axis along which to perform the EMA.

    Returns:
        Array with the same shape as x, containing the EMA along the specified axis.
    """

    # assert init_window_size > 0, "init_window_size must be greater than 0"

    alpha = jnp.clip(alpha, 0, 1)  # type: ignore

    # Take the average of the first `init_window_size` values as the initial EMA
    init_value = jnp.mean(jnp.take(x, jnp.arange(init_window_size), axis=axis), axis=axis)

    # Move the axis we're operating on to be the first dimension
    x_moved = jnp.moveaxis(x, axis, 0)

    def scan_fn(carry, x_t):
        ema = (1 - alpha) * carry + alpha * x_t
        return ema, ema

    _, ema = jax.lax.scan(scan_fn, init_value, x_moved)

    # Move the axis back to its original position
    return jnp.moveaxis(ema, 0, axis)