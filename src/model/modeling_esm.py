import torch
import torch.nn as nn
from transformers import (AutoModel)
from transformers.models.esm.modeling_esm import EsmEncoder


class MyEsmAutoModel(AutoModel):

    def __init__(self, config):
        super().__init__(config)
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, EsmEncoder):
            module.gradient_checkpointing = value