"""
implements PrGen class as described in the paper
returns log probability of prompt given passage
"""

# base class
from fitness_functions.PrGen import PrGen

# other imports
import numpy as np
from scipy.special import log_softmax

class PaperPrGen(PrGen):

    # p -> PrGen(p) = Pr[p | LAp]
    def evaluate(self, prompt, passage):

        # Pr[p]
        log_correct_p = self.get_ll(prompt + passage)

        # Pr[null prompt]
        log_null_p = self.get_ll(passage)

        # compares probabilities
        log_pr_correct = log_softmax(np.array([log_correct_p, log_null_p]))[0]

        # returns log probability
        return log_pr_correct

        



    
