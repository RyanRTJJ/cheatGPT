"""
prompt-free utility fn
uses prompt-agnostic components (f)
"""

# imports
import numpy as np
from utility_functions.PAUtilityFunction import PAUtilityFunction

class PFUF(PAUtilityFunction):

    # eval.s utility fn prompt-free
    def PA_u(self, p):

        Lp_batch = self.L.generate_batch(p, self.NUM_SAMPLES)
        DLp_batch = self.D.discriminate_batch(Lp_batch)
        fLp_batch = self.f.PA_evaluate_batch(Lp_batch)

        # synthesize D, f scores
        utility_batch = np.vectorize(self.synthesizer)(DLp_batch, fLp_batch)
        return np.average(utility_batch)
