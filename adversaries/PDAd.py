"""
Prompt-Default Adversary
given p_0 upon initialization
"""

# imports
import numpy as np
from adversaries.PAdversary import PAdversary

# defines PDAd class
class PDAd(PAdversary):

    # initializes PDAd
    def __init__(self, utility_function, p_0):
        super().__init__(utility_function)
        self.p_0 = p_0

    # defines train, test in terms of p_0
    # def PA_train(self):
    #     return self.train(self.p_0)
    
    def PA_test(self):
        return self.test(self.p_0)