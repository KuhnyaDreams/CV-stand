"""
Wrapper module for adversarial attack evaluation.
Provides attack implementations, evaluation framework, and utility functions.
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('wrapper.log', mode='a')
    ]
)

# Set root logger level
logger = logging.getLogger(__name__)
logger.info("Wrapper module initialized")

# Version
__version__ = "2.0.0"
__author__ = "CV Stand Team"

# Public API
from config import get_config, load_config, configure_logging
from attack_eval import AttackEvaluator, AttackExecutor
from bb_attacks import BlackBoxAttacks
from wb_attacks import WhiteBoxAttacks
from defense import Defenses, DefensePipeline
from model_functions import detect, estimate, segment, classify
from path_utils import PathManager
from config_validator import ConfigValidator

__all__ = [
    'get_config',
    'load_config',
    'configure_logging',
    'AttackEvaluator',
    'AttackExecutor',
    'BlackBoxAttacks',
    'WhiteBoxAttacks',
    'Defenses',
    'DefensePipeline',
    'detect',
    'estimate',
    'segment',
    'classify',
    'PathManager',
    'ConfigValidator',
]
