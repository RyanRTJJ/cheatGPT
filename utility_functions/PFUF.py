"""
prompt-free utility fn
uses prompt-agnostic components (f)

PROBABLY SHOULDN'T BE USED
PDUF w null prompt probably suffices
j call prompt-specific ff methods;
they'll auto-call relevant prompt-agnostic methods
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

    # interpretable version - returns metadata
    def PA_u_interpretable(self, p):

        Lp_batch = self.L.generate_batch(p, self.NUM_SAMPLES)
        DLp_batch = self.D.discriminate_batch(Lp_batch)
        fLp_batch = self.f.PA_evaluate_batch(Lp_batch)

        # synthesize D, f scores
        utility_batch = np.vectorize(self.synthesizer)(DLp_batch, fLp_batch)

        # returns everything!
        return {
            "prompt": p,
            "Lp_batch": Lp_batch,
            "DLp_batch": DLp_batch,
            "fLp_batch": fLp_batch,
            "utility_batch": utility_batch,
        }

