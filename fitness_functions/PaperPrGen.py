"""
implements PrGen class as described in the paper
returns probability to a constant exponent to avoid underflow
"""

# base class
from fitness_functions.PrGen import PrGen

# other imports
import numpy as np
from scipy.special import log_softmax

class PaperPrGen(PrGen):

    # scaling exponent
    LAMBDA = 0.1

    # p -> PrGen(p) = Pr[p | LAp]
    def evaluate(self, prompt, passage):

        # Pr[p]
        log_correct_p = self.get_ll(prompt + passage)

        # Pr[null prompt]
        log_null_p = self.get_ll(passage)

        # compares probabilities
        log_pr_correct = log_softmax(np.array([log_correct_p, log_null_p]))[0]

        # scales by exponent
        return np.exp(self.LAMBDA * log_pr_correct)



    
