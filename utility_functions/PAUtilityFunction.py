"""
prompt-agnostic utility function
prompt either hard-coded or uses prompt-agnostic components (f)
"""

# imports
import numpy as np
from utility_functions.UtilityFunction import UtilityFunction
from utility_functions.synthesis_functions import DEFAULT_SYNTHESIS_FN

class PAUtilityFunction(UtilityFunction):

    # template method
    def PA_u(self, p): raise NotImplementedError