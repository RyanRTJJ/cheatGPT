"""
PAdversary: Prompt-Agnostic Adversary
doesn't know p_0 when training
same prompt-default vs prompt-free split as PAUtilityFunction
"""

# imports
import numpy as np
from adversaries.Adversary import Adversary

# defines PAdversary class
class PAdversary(Adversary):

    # defines prompt-agnostic test stub
    def PA_test(self):
        raise NotImplementedError
    