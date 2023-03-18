"""
defines the `utility function` to be optimized by RL agents
synthesizes the scores of the discriminator and fitness function
objective function in approaches 1, 2
"""

# imports
import numpy as np
from utility_functions.synthesis_functions import DEFAULT_SYNTHESIS_FN

class UtilityFunction():

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

    # p -> u
    # fn of target prompt p_0
    def u(self, p, p_0):

        Lp_batch = self.L.generate_batch(p, self.NUM_SAMPLES)
        DLp_batch = self.D.discriminate_batch(Lp_batch)
        fLp_batch = self.f.evaluate_batch(p, Lp_batch)

        # synthesize D, f scores
        utility_batch = np.vectorize(self.synthesizer)(DLp_batch, fLp_batch)
        return np.average(utility_batch)