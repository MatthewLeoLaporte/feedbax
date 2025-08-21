"""Custom mapping classes.

:copyright: Copyright 2023-2024 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable, MutableMapping, Hashable

import dis
import logging
from typing import Any, Generic, TypeVar, overload

import equinox as eqx
# from equinox._pretty_print import tree_pp, bracketed
import jax
import jax.tree_util as jtu
import jax.tree as jt
from jaxtyping import Array, PyTree

from feedbax.misc import unzip2, where_func_to_attr_str_tree


logger = logging.getLogger(__name__)

T = TypeVar("T")
KT1 = TypeVar("KT1")
KT2 = TypeVar("KT2")
VT = TypeVar("VT")


class AbstractTransformedOrderedDict(MutableMapping[KT2, VT], Generic[KT1, KT2, VT]):
    """Base for `OrderedDict`s which transform keys when getting and setting items.

    It stores the original keys, and otherwise behaves (e.g. when iterating)
    as though they are the true keys.

    This is useful when we want to use a certain type of object as a key, but
    it would not be hashed properly by `OrderedDict`, so we need to transform
    it into something else. In particular, by replacing `_key_transform` with
    a code parsing function, a subclass of this class can take lambdas as keys.

    See [`feedbax.WhereDict`][feedbax.WhereDict] for an example.

    Based on https://stackoverflow.com/a/3387975
    """
    store: OrderedDict[KT1, tuple[KT2, VT]]

    def __init__(self, *args, **kwargs):
        self.store = OrderedDict()
        self.update(OrderedDict(*args, **kwargs))

    def __getitem__(self, key: KT1 | KT2) -> VT:
        k = self._key_transform(key)
        return self.store[k][1]

    @overload
    def get(self, key: KT1 | KT2) -> VT | None: ...

    @overload
    def get(self, key: KT1 | KT2, default: T) -> VT | T: ...

    def get(self, key, default=None):
        k = self._key_transform(key)
        if k in self.store:
            return self.store[k][1]
        else:
            return default

    def __setitem__(self, key: KT2, value: VT):
        self.store[self._key_transform(key)] = (key, value)

    def __delitem__(self, key: KT2):
        del self.store[self._key_transform(key)]

    def __iter__(self):
        for key in self.store:
            yield self.store[key][0]

    def __len__(self) -> int:
        return len(self.store)

    @abstractmethod
    def _key_transform(self, key: KT1 | KT2) -> KT1:
        ...

    def tree_flatten(self):
        """The same flatten function used by JAX for `dict`"""
        return tuple(self.values()), tuple(self.keys())

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))


class _QuotelessStr(str):
    """String that renders without quotation marks."""
    def __repr__(self):
        return self


class _WhereRepr:

    def __init__(self, where: Callable):
        bound_vars = dis.Bytecode(where).codeobj.co_varnames
        if len(bound_vars) != 1:
            raise ValueError("`WhereDict` keys must be functions of a single argument")

        self.bound_var = bound_vars[0]
        self.term_strs = where_func_to_attr_str_tree(where)

    def __repr__(self):
        terms = jt.map(
            lambda leaf: ".".join([self.bound_var, leaf]) if leaf else self.bound_var,
            self.term_strs,
        )
        # Wrap in parentheses so that the dict colon `:` is easier to distinguish from
        # the lambda colon.
        return f"lambda {self.bound_var}: {terms}"


T = TypeVar('T')


def _where_to_str(where: Callable) -> str:
    """Return a single string representing a `where` function."""
    terms = where_func_to_attr_str_tree(where)
    if isinstance(terms, str):
        where_str = terms
    else:
        where_str = ", ".join(jtu.tree_leaves(terms))
    return where_str


def _wheredict_key_repr(where_key: Callable | tuple[Callable, str]):
    """Return a string representation of a `WhereDict` key, which may be a function or a tuple."""
    if isinstance(where_key, Callable):
        where_repr = f"({_WhereRepr(where_key)})"
    elif isinstance(where_key, tuple):
        where_repr = f"(({_WhereRepr(where_key[0])}), \"{where_key[1]}\")"
    return _QuotelessStr(where_repr)


@jtu.register_pytree_node_class
class WhereDict(AbstractTransformedOrderedDict[str, Callable[[PyTree], Any], T]):
    """An `OrderedDict` that allows use of where-functions as keys.

    In particular, keys can be functions/lambdas that take a single argument,
    and return a single (nested) attribute accessed from that argument.

    Functions are parsed to equivalent strings, which can be used
    interchangeably as keys. For example, the following are equivalent when
    `init_spec` is a `WhereDict`:

    ```python
    init_spec[lambda state: state.mechanics.effector]
    ```
    ```python
    init_spec['mechanics.effector']
    ```

    Finally, a `tuple[Callable, str]` may also be provided as a key, for cases where
    different unique entries must be included for the same `Callable`. For example,
    the following are equivalent:

    ```python
    init_spec[(lambda state: state.mechanics.effector, "first")]
    ```
    ```python
    init_spec['mechanics.effector#first']
    ```

    Note that the hash symbol `#` is used to concatenate the usual string representation
    for the callable, with the paired string.

    ??? Note "Performance"
        For typical initialization mappings (1-10 items) construction is on the order
        of 50x slower than `OrderedDict`. Access is about 2-20x slower, depending
        whether indexed by string or by callable.

        However, we only need to do a single construction and a single access of
        `init_spec` per batch/evaluation, so performance shouldn't matter too much in
        practice: the added overhead is <50 us/batch, and a batch normally takes
        at least 20,000 us to train.
    """

    def _key_transform(self, key: str | Callable | tuple[Callable, str]) -> str:
        return self.key_transform(key)

    @staticmethod
    def key_transform(key: str | Callable | tuple[Callable, str]) -> str:

        if isinstance(key, str):
            pass
        elif isinstance(key, Callable):
            where_str = _where_to_str(key)
            return where_str
        elif isinstance(key, tuple):
            if not isinstance(key[0], Callable) or not isinstance(key[1], str):
                raise ValueError("Each `WhereDict` key should be supplied as a string, "
                                 "a callable, or a tuple of a callable and a string")
            where_str = _where_to_str(key[0])
            return '#'.join([where_str, key[1]])
        else:
            raise ValueError("Each `WhereDict` key should be supplied as a string, "
                             "a callable, or a tuple of a callable and a string")
        return key

    def _repr_with_indent(self, level: int) -> str:
        cls_name       = self.__class__.__name__
        current_indent = "  " * level
        item_indent    = "  " * (level + 1)

        # empty case
        if not self.store:
            return f"{cls_name}({{}})"

        item_strings = []
        for key, value in self.items():
            # turn your callable/tuple into a clean key repr (no extra quotes)
            key_repr = repr(_wheredict_key_repr(key))

            # capture repr(value) and split into lines
            raw_lines = repr(value).splitlines() or [""]
            if len(raw_lines) > 1:
                # first line as‐is, indent subsequent
                first, *rest = raw_lines
                rest_indented = "\n".join(f"{item_indent}{ln}" for ln in rest)
                value_repr = f"{first}\n{rest_indented}"
            else:
                value_repr = raw_lines[0]

            item_strings.append(f"{item_indent}{key_repr}: {value_repr},")

        body = "\n".join(item_strings)
        return (
            f"{cls_name}({{\n"
            f"{body}\n"
            f"{current_indent}}})"
        )

    def __repr__(self) -> str:
        # now just forward to your own indent‐printer
        return self._repr_with_indent(0)



