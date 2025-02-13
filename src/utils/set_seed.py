# -------- Done By Manu Bhaskar ---------

# -------- Dependencies ------------
from src.utils.logger import LoggerSetup
from config.config import Config
from transformers import set_seed
import numpy as np
import random
import torch
import os

def set_global_seed(logger: LoggerSetup ,seed: int = Config.RANDOM_SEED) -> None:
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed=seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set as {seed}")
