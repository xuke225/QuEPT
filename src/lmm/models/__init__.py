import importlib
import os
import sys

import hf_transfer
from loguru import logger

from models.internvl2 import InternVL2
from models.llava_onevision import LLaVA_onevision
from models.llava_v15 import LLaVA_v15
from models.qwen2_vl import Qwen2_VL
from models.vila import vila

from utils.registry import MODEL_REGISTRY

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

def get_process_model(model_name):
    return MODEL_REGISTRY[model_name]