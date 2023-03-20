"""
prompt-determined RLHF
optimizes over utility given a fixed prompt
uses trlx to fine-tune a LM to produce prompts
to optimize utility via RLHF
"""

# imports
from adversaries.Adversary import Adversary
from TIH import trlx

class PD_RLHF(Adversary):
    pass
