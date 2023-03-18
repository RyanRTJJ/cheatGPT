"""
prompt-default utility fn
uses default prompt p_0 to compute utility
"""

# imports
import numpy as np
from utility_functions.PAUtilityFunction import PAUtilityFunction

class PDUF(PAUtilityFunction):

    # takes in prompt p_0
    def __init__(self, L, D, f, p_0):
        # calls superclass constructor
        super().__init__(L, D, f)

        # sets default prompt
        self.p_0 = p_0

    # eval.s utility fn using default prompt
    def PA_u(self, p):
        return self.evaluate(p, self.p_0)
