"""
defines the `utility function` to be optimized by RL agents
synthesizes the scores of the discriminator and fitness function
objective function in approaches 1, 2
"""

# imports
import numpy as np
from utility_functions.synthesis_functions import DEFAULT_SYNTHESIS_FN

class UtilityFunction:

    # default number of samples p -> L(p) taken by utility fn
    NUM_SAMPLES = 10

    # initializes w D, f
    def __init__(self, L, D, f, synthesizer = DEFAULT_SYNTHESIS_FN):

        # sets generator
        self.L = L

        # sets discriminator
        self.D = D

        # sets fitness function
        self.f = f

        # sets synthesis function
        self.synthesizer = synthesizer

    # interpretable version - returns metadata
    def u_interpretable(self, Ap, p):
            
        LAp_batch = self.L.generate_batch(Ap, self.NUM_SAMPLES)
        DLAp_batch = self.D.discriminate_batch(LAp_batch)
        fLAp_batch = self.f.evaluate_batch(p, LAp_batch)

        # synthesize D, f scores
        utility_batch = np.vectorize(self.synthesizer)(DLAp_batch, fLAp_batch)

        # returns everything!
        return {
            "prompt": Ap,
            "Lp_batch": LAp_batch,
            "DLp_batch": DLAp_batch,
            "fLp_batch": fLAp_batch,
            "utility_batch": utility_batch,
        }
    
    # Ap -> u
    # fn of target prompt p
    def u(self, Ap, p):

        # calls u_interpretable
        utility_batch = self.u_interpretable(Ap, p)["utility_batch"]
        return np.average(utility_batch)
    