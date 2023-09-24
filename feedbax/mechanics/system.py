"""

:copyright: Copyright 2023 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

import logging 
from typing import Any, Protocol, TypeVar

from jaxtyping import PyTree


logger = logging.getLogger(__name__)


T = TypeVar("T")


# TODO maybe this should be `AbstractSystem(eqx.Module)` instead
class System(Protocol):
    def vector_field(
        self, 
        t: float, 
        y: PyTree[T], 
        args: PyTree,  # controls
) -> PyTree[T]:
        """Vector field of the system."""
        pass

    @property
    def control_size(self) -> int:
        """Number of control inputs."""
        pass