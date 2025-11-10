"""Neural network architectures.

:copyright: Copyright 2023-2024 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

import logging
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from typing import (
    Literal,
    Optional,
    Protocol,
    Self,
    Type,
    Union,
    runtime_checkable,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import Module, field
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax._model import wrap_stateless_callable, wrap_stateless_keyless_callable
from feedbax._staged import AbstractStagedModel, ModelStage
from feedbax.intervene import AbstractIntervenor
from feedbax.intervene.schedule import ArgIntervenors, ModelIntervenors
from feedbax.misc import (
    identity_func,
    interleave_unequal,
    n_positional_args,
)
from feedbax.state import StateT

logger = logging.getLogger(__name__)


# class Layer(Protocol):
#     def __init__(
#         self,
#         input_size: int,
#         hidden_size: int,
#         use_bias: bool = True,
#         *,
#         key: PRNGKeyArray,
#         **kwargs,
#     ): ...

#     def __call__(
#         self,
#         input: Array,
#         state: Array,
#         *,
#         key: PRNGKeyArray,
#     ) -> Array: ...


def orthogonal_gru_cell(
    input_size: int,
    hidden_size: int,
    use_bias: bool = True,
    scale: float = 1.0,
    *,
    key: PRNGKeyArray,
):
    """Returns an `eqx.nn.GRUCell` with orthogonal weight matrix initialization."""
    net = eqx.nn.GRUCell(input_size, hidden_size, use_bias=use_bias, key=key)
    initializer = jax.nn.initializers.orthogonal(scale=scale, column_axis=-1)
    ortho_weight_hh = jnp.concatenate(
        [initializer(k, (hidden_size, hidden_size)) for k in jr.split(key, 3)],
        axis=0,
    )
    net = eqx.tree_at(
        lambda net: net.weight_hh,
        net,
        ortho_weight_hh,
    )
    return net


class NetworkState(Module):
    """Type of state PyTree operated on by [`SimpleStagedNetwork`][feedbax.nn.SimpleStagedNetwork] instances.

    Attributes:
        hidden: The (output) activity of the hidden layer units.
        output: The activity of the readout layer, if the network has one.
        encoding: The activity of the encoding layer, if the network has one.
    """

    input: Float[Array, "inputs"]
    hidden: PyTree[Float[Array, "unit"]]
    output: Optional[PyTree[Array]] = None
    encoding: Optional[PyTree[Array]] = None


def contiguous_assignment(
    hidden_size: int,
    n_input_only: int,
    n_readout_only: int,
    n_recurrent_only: int,
    n_input_readout: int,
    key: PRNGKeyArray,
):
    """Contiguous assignment function for population structure.

    Assigns populations in contiguous blocks:
    - Units 0 to n_input_only-1: input-only
    - Units n_input_only to n_input_only+n_readout_only-1: readout-only
    - Units ... to ...: recurrent-only
    - Remaining units: input-readout

    Arguments:
        hidden_size: Total number of hidden units.
        n_input_only: Number of input-only units.
        n_readout_only: Number of readout-only units.
        n_recurrent_only: Number of recurrent-only units.
        n_input_readout: Number of input-readout units.
        key: Random key (unused, but required for interface compatibility).

    Returns:
        Tuple of (input_only_indices, readout_only_indices, recurrent_only_indices, input_readout_indices).
    """
    input_only_indices = jnp.arange(0, n_input_only)
    readout_only_indices = jnp.arange(n_input_only, n_input_only + n_readout_only)
    recurrent_only_indices = jnp.arange(
        n_input_only + n_readout_only,
        n_input_only + n_readout_only + n_recurrent_only,
    )
    input_readout_indices = jnp.arange(
        n_input_only + n_readout_only + n_recurrent_only,
        hidden_size,
    )
    return input_only_indices, readout_only_indices, recurrent_only_indices, input_readout_indices


class PopulationStructure(Module):
    """Defines partitioning of hidden units into subpopulations with different connectivity.

    Hidden units can be partitioned into four types:
    - Input-only: receive inputs but don't contribute to readout
    - Readout-only: don't receive inputs but contribute to readout
    - Recurrent-only: neither receive inputs nor contribute to readout (internal dynamics only)
    - Input-readout: both receive inputs and contribute to readout

    Attributes:
        n_input_only: Number of units that only receive inputs.
        n_readout_only: Number of units that only contribute to readout.
        n_recurrent_only: Number of units that are recurrent-only (no input/output).
        n_input_readout: Number of units that both receive inputs and contribute to readout.
        input_indices: Indices of all units receiving inputs (input-only + input-readout).
        readout_indices: Indices of all units contributing to readout (readout-only + input-readout).
        input_only_indices: Indices of input-only units.
        readout_only_indices: Indices of readout-only units.
        recurrent_only_indices: Indices of recurrent-only units.
        input_readout_indices: Indices of input-readout units.
    """

    n_input_only: int
    n_readout_only: int
    n_recurrent_only: int
    n_input_readout: int

    # Indices for each population
    input_indices: Array  # shape (n_input_only + n_input_readout,)
    readout_indices: Array  # shape (n_readout_only + n_input_readout,)
    input_only_indices: Array  # shape (n_input_only,)
    readout_only_indices: Array  # shape (n_readout_only,)
    recurrent_only_indices: Array  # shape (n_recurrent_only,)
    input_readout_indices: Array  # shape (n_input_readout,)

    @classmethod
    def create(
        cls,
        hidden_size: int,
        n_input_only: int = 0,
        n_readout_only: int = 0,
        n_recurrent_only: int = 0,
        n_input_readout: int = 0,
        assignment_fn: Optional[Callable] = None,
        *,
        key: PRNGKeyArray,
    ) -> "PopulationStructure":
        """Create a population structure with the specified population sizes.

        If the subpopulation sizes (e.g. `n_input_only`) sum up to less than
        the `hidden_size`, then `n_input_readout` will be increased so the
        overall sizes match. That means that calling this method with only
        a `hidden_size` will create a network with no input-output segregation.
        (Alternatively, in the future we could ask only for the subpopulation
        structure and not for the total `hidden_size`, and just infer it.)

        Arguments:
            hidden_size: Total number of hidden units.
            n_input_only: Number of input-only units.
            n_readout_only: Number of readout-only units.
            n_recurrent_only: Number of recurrent-only units.
            n_input_readout: Number of input-readout units.
            assignment_fn: Optional callable that takes (hidden_size, n_input_only,
                n_readout_only, n_recurrent_only, n_input_readout, key) and returns
                a tuple of 4 arrays (input_only_indices, readout_only_indices,
                recurrent_only_indices, input_readout_indices). If None, uses random
                assignment.
            key: Random key for assignment.
        """
        total = n_input_only + n_readout_only + n_recurrent_only + n_input_readout
        if total > hidden_size:
            raise ValueError("Sum of subpopulation sizes is larger than network size")
        elif total < hidden_size:
            n_input_readout += hidden_size - total

        if assignment_fn is None:
            # Default: random assignment
            all_indices = jr.permutation(key, hidden_size)
            input_only_indices = all_indices[:n_input_only]
            readout_only_indices = all_indices[n_input_only : n_input_only + n_readout_only]
            recurrent_only_indices = all_indices[
                n_input_only + n_readout_only : n_input_only + n_readout_only + n_recurrent_only
            ]
            input_readout_indices = all_indices[n_input_only + n_readout_only + n_recurrent_only :]
        else:
            # Custom assignment
            (
                input_only_indices,
                readout_only_indices,
                recurrent_only_indices,
                input_readout_indices,
            ) = assignment_fn(
                hidden_size, n_input_only, n_readout_only, n_recurrent_only, n_input_readout, key
            )

        # Compute combined indices for input and readout
        input_indices = jnp.concatenate([input_only_indices, input_readout_indices])
        readout_indices = jnp.concatenate([readout_only_indices, input_readout_indices])

        return cls(
            n_input_only=n_input_only,
            n_readout_only=n_readout_only,
            n_recurrent_only=n_recurrent_only,
            n_input_readout=n_input_readout,
            input_indices=input_indices,
            readout_indices=readout_indices,
            input_only_indices=input_only_indices,
            readout_only_indices=readout_only_indices,
            recurrent_only_indices=recurrent_only_indices,
            input_readout_indices=input_readout_indices,
        )


class MaskedLinear(Module):
    """A linear layer with a fixed mask enforcing structural zeros.

    This layer wraps an eqx.nn.Linear layer and applies a binary mask to enforce
    that certain connections remain zero throughout training.

    Attributes:
        linear: The underlying linear layer.
        mask: Binary mask applied to weights (1 = trainable, 0 = structural zero).
    """

    linear: eqx.nn.Linear
    mask: Array  # shape matches linear.weight

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: Array,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize a masked linear layer.

        Arguments:
            in_features: Number of input features.
            out_features: Number of output features.
            mask: Binary mask of shape (out_features, in_features). 1 = trainable, 0 = always zero.
            use_bias: Whether to include a bias term.
            key: Random key for weight initialization.
        """
        self.linear = eqx.nn.Linear(in_features, out_features, use_bias=use_bias, key=key)
        self.mask = mask

    @property
    def weight(self) -> Array:
        """Access the underlying weight matrix for compatibility with eqx.nn.Linear interface."""
        return self.linear.weight

    @property
    def bias(self) -> Optional[Array]:
        """Access the underlying bias for compatibility with eqx.nn.Linear interface."""
        return self.linear.bias

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """Apply the masked linear transformation.

        The mask is applied on every forward pass to ensure structural zeros.

        Arguments:
            x: Input array.
            key: Optional random key (unused, but required for compatibility).
        """
        # Apply mask to weights before computation
        masked_weight = self.linear.weight * self.mask
        result = jnp.dot(x, masked_weight.T)
        if self.linear.bias is not None:
            result = result + self.linear.bias
        return result


class SimpleStagedNetwork(AbstractStagedModel[NetworkState]):
    """A single step of a neural network layer, with optional encoder and readout layers.

    Attributes:
        hidden_size: The number of units in the hidden layer.
        out_size: The number of readout units, if the network has a readout layer. Otherwise
            this is equal to `hidden_size`.
        encoding_size: The number of encoder units, if the network has an encoder layer.
        hidden: The module implementing the hidden layer.
        hidden_nonlinearity: The nonlinearity applied to the hidden layer output.
        encoder: The module implementing the encoder layer, if present.
        readout: The module implementing the readout layer, if present.
    """

    input_size: int
    hidden: Module
    hidden_size: int
    hidden_noise_std: Optional[float]
    hidden_nonlinearity: Callable[[Float], Float]
    out_size: int
    out_nonlinearity: Callable[[Float], Float]
    readout: Optional[Module] = None
    encoding_size: Optional[int] = None
    encoder: Optional[Module] = None
    population_structure: Optional[PopulationStructure] = None

    intervenors: ModelIntervenors[NetworkState]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        out_size: Optional[int] = None,
        encoding_size: Optional[int] = None,
        hidden_type: Callable[..., Module] = eqx.nn.GRUCell,
        encoder_type: Callable[..., Module] = eqx.nn.Linear,
        readout_type: Callable[..., Module] = eqx.nn.Linear,
        use_bias: bool = True,
        hidden_nonlinearity: Callable[[Float], Float] = identity_func,
        out_nonlinearity: Callable[[Float], Float] = identity_func,
        hidden_noise_std: Optional[float] = None,
        population_structure: Optional[PopulationStructure] = None,
        intervenors: Optional[ArgIntervenors] = None,
        *,
        key: PRNGKeyArray,
    ):
        """
        !!! Note
            If an integer is passed for `encoding_size`, input encoding is enabled.
            Otherwise network inputs are passed directly to the hidden layer.

            If an integer is passed for `out_size`, readout is enabled. Otherwise
            the network's outputs are the outputs of the hidden units.

            In principle `hidden_type` can be class defining a multi-layer network,
            as long as it is instantiated as `hidden_type(input_size, hidden_size, use_bias, *,
            key)`.

            Use `partial` to set `use_bias` for the encoder or readout types, before
            passing them to this constructor.

        ??? dev-note
            It is difficult to type check `hidden_type`, since there is no superclass
            for stateful network layers, and protocols do not work with `eqx.Module`.
            Currently we do not check its signatures, or the signatures of the
            callable it returns. This means it is up to the user to supply the right
            kind of class here, which we have to document.

            Perhaps it is unwise to construct `SimpleStagedNetwork` as it is.
            Maybe layers should be kept separate, and individually added by the user.
            That would avert the relatively complicated logic in `model_spec`, here.

        Arguments:
            input_size: The number of input channels in the network.
                If `encoder_type` is not `None`, this is the number of inputs
                to the encoder layer—otherwise, the hidden layer.
            hidden_size: The number of units in the hidden layer.
            out_size: The number of readout units. If `None`, do not add a readout layer.
            encoding_size: The number of encoder units. If `None`, do not add an encoder layer.
            hidden_type: The type of hidden layer to use.
            encoder_type: The type of encoder layer to use.
            use_bias: Whether the hidden layer should have a bias term.
            hidden_nonlinearity: A function to apply unitwise to the hidden layer output. This is
                typically not used if `hidden_type` is `GRUCell` or `LSTMCell`.
            out_nonlinearity: A function to apply unitwise to the readout layer output.
            hidden_noise_std: Standard deviation of Gaussian noise to add to the hidden layer output.
            population_structure: Optional population structure defining which hidden units
                receive inputs and/or contribute to readout. If provided, input and readout
                weights will be masked to enforce the specified connectivity pattern.
            intervenors: [Intervenors][feedbax.intervene.AbstractIntervenor] to add
                to the model at construction time.
            key: Random key for initialising the network.
        """
        key1, key2, key3 = jr.split(key, 3)

        self.input_size = input_size
        self.population_structure = population_structure

        # Create encoder layer (potentially masked if population_structure is provided)
        if encoding_size is not None:
            if population_structure is not None:
                # Create mask for encoder: only input-receiving units get non-zero columns
                encoder_mask = jnp.zeros((encoding_size, input_size))
                # For simplicity, allow all encoder units to receive all inputs
                # The masking will happen at the encoder->hidden connection instead
                encoder_mask = jnp.ones((encoding_size, input_size))
                self.encoder = MaskedLinear(
                    input_size, encoding_size, encoder_mask, use_bias=use_bias, key=key2
                )
            else:
                self.encoder = encoder_type(input_size, encoding_size, key=key2)
            self.encoding_size = encoding_size

            # Create hidden layer - if we have population structure, we need to mask
            # the connection from encoder to hidden
            if population_structure is not None:
                # For RNN cells like GRUCell, we need to mask the input weights
                # Create the hidden layer first, then mask its input weights
                hidden = hidden_type(encoding_size, hidden_size, use_bias=use_bias, key=key1)

                # Mask the input->hidden weights (weight_ih for GRU/RNN)
                if hasattr(hidden, "weight_ih"):
                    # For GRUCell, weight_ih is (3*hidden_size, input_size) for reset, update, candidate
                    weight_ih_shape = hidden.weight_ih.shape

                    # Create mask: only input-receiving units get non-zero rows
                    if weight_ih_shape[0] == 3 * hidden_size:
                        # GRUCell case: replicate mask 3 times (for reset, update, candidate)
                        hidden_input_mask = jnp.zeros((hidden_size, encoding_size))
                        hidden_input_mask = hidden_input_mask.at[
                            population_structure.input_indices, :
                        ].set(1.0)
                        hidden_input_mask = jnp.tile(hidden_input_mask, (3, 1))
                    else:
                        # Simple RNN case
                        hidden_input_mask = jnp.zeros((hidden_size, encoding_size))
                        hidden_input_mask = hidden_input_mask.at[
                            population_structure.input_indices, :
                        ].set(1.0)

                    masked_weight_ih = hidden.weight_ih * hidden_input_mask
                    hidden = eqx.tree_at(lambda h: h.weight_ih, hidden, masked_weight_ih)

                self.hidden = hidden
            else:
                self.hidden = hidden_type(encoding_size, hidden_size, use_bias=use_bias, key=key1)
        else:
            # No encoder - input goes directly to hidden layer
            if population_structure is not None:
                hidden = hidden_type(input_size, hidden_size, use_bias=use_bias, key=key1)

                # Mask the input->hidden weights
                if hasattr(hidden, "weight_ih"):
                    # For GRUCell, weight_ih is (3*hidden_size, input_size) for reset, update, candidate
                    weight_ih_shape = hidden.weight_ih.shape

                    # Create mask: only input-receiving units get non-zero rows
                    if weight_ih_shape[0] == 3 * hidden_size:
                        # GRUCell case: replicate mask 3 times (for reset, update, candidate)
                        hidden_input_mask = jnp.zeros((hidden_size, input_size))
                        hidden_input_mask = hidden_input_mask.at[
                            population_structure.input_indices, :
                        ].set(1.0)
                        hidden_input_mask = jnp.tile(hidden_input_mask, (3, 1))
                    else:
                        # Simple RNN case
                        hidden_input_mask = jnp.zeros((hidden_size, input_size))
                        hidden_input_mask = hidden_input_mask.at[
                            population_structure.input_indices, :
                        ].set(1.0)

                    masked_weight_ih = hidden.weight_ih * hidden_input_mask
                    hidden = eqx.tree_at(lambda h: h.weight_ih, hidden, masked_weight_ih)

                self.hidden = hidden
            else:
                self.hidden = hidden_type(input_size, hidden_size, use_bias=use_bias, key=key1)

        self.hidden_size = hidden_size
        self.hidden_nonlinearity = hidden_nonlinearity
        self.hidden_noise_std = hidden_noise_std

        # Create readout layer (potentially masked if population_structure is provided)
        if out_size is not None:
            if population_structure is not None:
                # Create mask for readout: only readout-contributing units have non-zero columns
                readout_mask = jnp.zeros((out_size, hidden_size))
                readout_mask = readout_mask.at[:, population_structure.readout_indices].set(1.0)
                readout = MaskedLinear(
                    hidden_size, out_size, readout_mask, use_bias=use_bias, key=key3
                )
            else:
                readout = readout_type(hidden_size, out_size, key=key3)

            if (bias := getattr(readout, "bias", None)) is not None:
                if isinstance(readout, MaskedLinear):
                    #! Is this necessary? Or can we just tree_at `layer.bias` since it is a property
                    #! that points to `layer.linear.bias`?
                    readout = eqx.tree_at(
                        lambda layer: layer.linear.bias,
                        readout,
                        jnp.zeros_like(bias),
                    )
                else:
                    readout = eqx.tree_at(
                        lambda layer: layer.bias,
                        readout,
                        jnp.zeros_like(bias),
                    )
            self.readout = readout
            self.out_nonlinearity = out_nonlinearity
            self.out_size = out_size
        else:
            self.out_size = hidden_size

        self.intervenors = self._get_intervenors_dict(intervenors)

    def _add_hidden_noise(self, input, state, *, key):
        if self.hidden_noise_std is None:
            return state
        return state + self.hidden_noise_std * jr.normal(key, state.shape)

    @property
    def model_spec(self) -> OrderedDict[str, ModelStage[Self, NetworkState]]:
        """Specifies the network model stages: layers, nonlinearities, and noise.

        Only includes stages for the encoding layer, readout layer, hidden noise, and
        hidden nonlinearity, if the user respectively requests them at the time of
        construction.

        !!! NOTE
            Inspects the instantiated hidden layer to determine if it is a stateful
            network (e.g. an RNN). If not (e.g. Linear), it wraps the layer so that
            it plays well with the state-passing of `AbstractStagedModel`. This assumes
            that stateful layers will take 2 positional arguments, and stateless layers
            only 1.
        """
        Stage = ModelStage[Self, NetworkState]

        if n_positional_args(self.hidden) == 1:  # type: ignore
            hidden_module = lambda self: wrap_stateless_callable(self.hidden)
            if isinstance(self.hidden, eqx.nn.Linear):
                logger.warning(
                    "Network hidden layer is linear but no hidden nonlinearity is defined"
                )
        else:
            # #TODO: revert this!
            # def tmp(self):
            #     def wrapper(input, state, *, key):
            #         return self.hidden(input, jnp.zeros_like(state))
            #     return wrapper
            # hidden_module = lambda self: tmp(self)
            hidden_module = lambda self: self.hidden

        spec = OrderedDict(
            {
                # Store the flattened network inputs as part of `NetworkState`
                "input": Stage(
                    callable=lambda self: lambda input, state, *, key: input,
                    where_input=lambda input, _: ravel_pytree(input)[0],
                    where_state=lambda state: state.input,
                ),
            }
        )

        if self.encoder is None:
            spec |= OrderedDict(
                {
                    "hidden": Stage(
                        callable=hidden_module,
                        where_input=lambda input, state: state.input,
                        where_state=lambda state: state.hidden,
                    ),
                }
            )
        else:
            spec = OrderedDict(
                {
                    "encoder": Stage(
                        callable=lambda self: lambda input, state, *, key: self.encoder(input),
                        where_input=lambda input, state: state.input,
                        where_state=lambda state: state.encoding,
                    ),
                    "hidden": Stage(
                        callable=hidden_module,
                        where_input=lambda input, state: state.encoding,
                        where_state=lambda state: state.hidden,
                    ),
                }
            )

        # TODO: conditional on `self.hidden_nonlinearity is not identity_func`?
        spec |= {
            "hidden_nonlinearity": Stage(
                callable=lambda self: wrap_stateless_keyless_callable(self.hidden_nonlinearity),
                where_input=lambda input, state: state.hidden,
                where_state=lambda state: state.hidden,
            ),
        }

        if self.hidden_noise_std is not None:
            spec |= {
                "hidden_noise": Stage(
                    callable=lambda self: self._add_hidden_noise,
                    where_input=lambda input, state: state.hidden,
                    where_state=lambda state: state.hidden,
                ),
            }

        if self.readout is not None:
            spec |= {
                "readout": Stage(
                    callable=lambda self: wrap_stateless_callable(self.readout),  # type: ignore
                    where_input=lambda input, state: state.hidden,
                    where_state=lambda state: state.output,
                ),
            }

        spec |= {
            "out_nonlinearity": Stage(
                callable=lambda self: wrap_stateless_keyless_callable(self.out_nonlinearity),
                where_input=lambda input, state: state.output,
                where_state=lambda state: state.output,
            )
        }

        return spec

    @property
    def memory_spec(self) -> PyTree[bool]:
        return NetworkState(
            input=True,
            hidden=True,
            output=True,
            encoding=True,
        )

    def init(self, *, key: PRNGKeyArray):
        if self.out_size is None:
            output = None
        else:
            output = jnp.zeros(self.out_size)

        if self.encoding_size is None:
            encoding = None
        else:
            encoding = jnp.zeros(self.encoding_size)

        # TODO: Try eval_shape
        return NetworkState(
            input=jnp.zeros(self.input_size),
            hidden=jnp.zeros(self.hidden_size),
            output=output,
            encoding=encoding,
        )


class LeakyRNNCell(Module):
    """Custom `RNNCell` with persistent, leaky state.

    Based on `eqx.nn.GRUCell` and the leaky RNN from

        [1] G. R. Yang, M. R. Joglekar, H. F. Song, W. T. Newsome,
            and X.-J. Wang, “Task representations in neural networks trained
            to perform many cognitive tasks,” Nat Neurosci, vol. 22, no. 2,
            pp. 297–306, Feb. 2019, doi: 10.1038/s41593-018-0310-2.
    """

    weight_hh: Array
    weight_ih: Array
    bias: Optional[Array]
    input_size: int
    hidden_size: int
    use_bias: bool
    use_noise: bool
    noise_strength: float
    dt: float
    tau: float
    nonlinearity: Callable

    @jax.named_scope("fbx.RNNCell")
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        use_noise: bool = False,
        noise_strength: float = 0.01,
        dt: float = 1.0,
        tau: float = 1.0,
        nonlinearity: Callable = jnp.tanh,
        *,  # this forces the user to pass the following as keyword arguments
        key: PRNGKeyArray,
        **kwargs,
    ):
        ihkey, hhkey, bkey = jr.split(key, 3)
        lim = math.sqrt(1 / hidden_size)

        if input_size > 0:
            self.weight_ih = jr.uniform(
                ihkey,
                (hidden_size, input_size),
                minval=-lim,
                maxval=lim,
            )
        else:
            self.weight_ih = jnp.array(0)

        self.weight_hh = jr.uniform(
            hhkey,
            (hidden_size, hidden_size),
            minval=-lim,
            maxval=lim,
        )

        if use_bias:
            self.bias = jr.uniform(
                bkey,
                (hidden_size,),
                minval=-lim,
                maxval=lim,
            )
        else:
            self.bias = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.use_noise = use_noise
        self.noise_strength = noise_strength
        self.dt = dt
        self.tau = tau
        self.nonlinearity = nonlinearity

    def __call__(self, input: Array, state: Array, key: PRNGKeyArray):
        """Vanilla RNN cell."""
        if self.use_bias:
            bias = self.bias
        else:
            bias = 0

        if self.use_noise:
            noise = self.noise_std * jr.normal(key, state.shape)
        else:
            noise = 0

        state = (1 - self.alpha) * state + self.alpha * self.nonlinearity(
            jnp.dot(self.weight_ih, input) + jnp.dot(self.weight_hh, state) + bias + noise
        )

        return state  #! 0D PyTree

    @cached_property
    def alpha(self):
        return self.dt / self.tau

    @cached_property  # type: ignore
    def noise_std(self, noise_strength):
        if self.use_noise:
            return math.sqrt(2 / self.alpha) * noise_strength
        else:
            return None


def n_layer_linear(
    hidden_sizes: Sequence[int],
    input_size: int,
    out_size: int,
    use_bias: bool = True,
    nonlinearity: Callable[[Float], Float] = jnp.tanh,
    *,
    key,
):
    """A simple n-layer linear network with nonlinearity."""
    keys = jr.split(key, len(hidden_sizes) + 1)
    sizes = (input_size,) + tuple(hidden_sizes) + (out_size,)
    layers = [
        eqx.nn.Linear(size0, size1, use_bias=use_bias, key=keys[i])
        for i, (size0, size1) in enumerate(zip(sizes[:-1], sizes[1:]))
    ]
    return eqx.nn.Sequential(list(interleave_unequal(layers, [nonlinearity] * len(hidden_sizes))))


def two_layer_linear(
    hidden_size, input_size, out_size, use_bias=True, nonlinearity=jnp.tanh, *, key
):
    """A two-layer linear network with nonlinearity.

    Just a convenience over `n_layer` since two-layer readouts may be more
    common than other n-layer readouts for RNNs.
    """
    return n_layer_linear(
        (hidden_size,),
        input_size,
        out_size,
        use_bias=use_bias,
        nonlinearity=nonlinearity,
        key=key,
    )


def gru_weight_idxs_func(
    label: Literal["candidate", "update", "reset"],
) -> Callable[[Array], slice]:
    """Returns a function that returns a slice of a subset of the GRU weights.

    TODO: Should probably just return a function that returns the subset of weights directly, rather than their indices.
    """

    def gru_weight_idxs(weights):
        len_by_3 = weights.shape[-2] // 3
        return {
            "reset": slice(0, len_by_3),
            "update": slice(len_by_3, 2 * len_by_3),
            "candidate": slice(2 * len_by_3, None),
        }[label]

    return gru_weight_idxs
