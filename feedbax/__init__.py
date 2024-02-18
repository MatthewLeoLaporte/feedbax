"""TODO

:copyright: Copyright 2023-2024 by Matt L Laporte.
:license: Apache 2.0, see LICENSE for details.
"""

import logging 
import logging.handlers as loghandlers
import os 
import warnings

from feedbax._io import save, load
from feedbax.model import get_ensemble
from feedbax._staged import (
    AbstractStagedModel, 
    ModelStage, 
    pformat_model_spec,
    pprint_model_spec,
)
from feedbax._tree import (
    random_split_like_tree,
    tree_array_bytes, 
    tree_call,
    tree_map_unzip,
    tree_set, 
    tree_stack,
    tree_struct_bytes,
    tree_take,
    tree_unzip,
)

# logging.config.fileConfig('../logging.conf')
LOG_LEVEL = os.environ.get('FEEDBAX_LOG_LEVEL', 'DEBUG').upper()

logger = logging.getLogger(__package__)
logger.setLevel(LOG_LEVEL)

file_handler = loghandlers.RotatingFileHandler(
    f'{__package__}.log',
    maxBytes=1_000_000,
    backupCount=1,
)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s,%(lineno)d: %(message)s",
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logging.captureWarnings(True)

logger.info('Logger configured.')


