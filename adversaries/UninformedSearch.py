"""
Uninformed Search Adversary
train: evalues all prompts in given list
test: chooses best prompt
prompt-free; just optimizes exogenous (prompt-agnostic) utility fn
"""

# imports
import numpy as np
from adversaries.PFAd import PFAd

# defines UninformedSearch class
class UninformedSearch(PFAd):

    # initializes w list of prompts
    def __init__(self, utility_function, prompts):
        super().__init__(utility_function)
        self.prompts = prompts

        # dummy var for best prompt
        self.best_prompt = None

    # trains adversary
    def train(self):

        # finds argmax of utility function over prompts
        utilities = [self.utility_function.PA_u(p) for p in self.prompts]
        self.best_prompt = self.prompts[np.argmax(utilities)]

    # returns best prompt
    def test(self):
        return self.best_prompt
