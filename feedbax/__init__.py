"""
:copyright: Copyright 2023-2024 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

import importlib.metadata
import logging
import logging.handlers as loghandlers
import os
import warnings


from feedbax._io import save, load, load_with_hyperparameters
from feedbax._model import (
    AbstractModel,
    ModelInput,
    wrap_stateless_callable,
    wrap_stateless_keyless_callable,
)
from feedbax._staged import (
    AbstractStagedModel,
    ModelStage,
    pformat_model_spec,
    pprint_model_spec,
)
from feedbax._tree import (
    get_ensemble,
    random_split_like_tree,
    is_type,
    leaves_of_type,
    make_named_dict_subclass,
    make_named_tuple_subclass,
    move_level_to_outside,
    tree_array_bytes,
    tree_call,
    tree_concatenate,
    tree_infer_batch_size,
    tree_key_tuples,
    tree_labels,
    tree_labels_of_equal_leaves,
    tree_map_tqdm,
    tree_map_unzip,
    tree_prefix_expand,
    tree_set,
    tree_set_scalar,
    tree_stack,
    tree_struct_bytes,
    tree_take,
    tree_take_multi,
    tree_unstack,
    tree_unzip,
    tree_zip,
)
from feedbax.intervene import is_intervenor
from feedbax.loss import is_lossdict
from feedbax.misc import is_module
# from feedbax._logging import enable_central_logging


__version__ = importlib.metadata.version("feedbax")


# logging.config.fileConfig('../logging.conf')

if os.environ.get("FEEDBAX_DEBUG", False) == "True":
    DEFAULT_LOG_LEVEL = "DEBUG"
else:
    DEFAULT_LOG_LEVEL = "INFO"

LOG_LEVEL = os.environ.get("FEEDBAX_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()


logger = logging.getLogger(__package__)
logger.addHandler(logging.NullHandler())

