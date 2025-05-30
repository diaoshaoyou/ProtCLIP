import enum
import peft
from peft import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING


# register MOLE
class PeftType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"
    IA3 = "IA3"
    MOLE = 'MOLE'

peft.PeftType = PeftType

from .mole import MOLEConfig, MOLEModel
PEFT_TYPE_TO_CONFIG_MAPPING[peft.PeftType.MOLE] = MOLEConfig
PEFT_TYPE_TO_MODEL_MAPPING[peft.PeftType.MOLE] = MOLEModel


__all__ = [
    'MOLEConfig', 
]