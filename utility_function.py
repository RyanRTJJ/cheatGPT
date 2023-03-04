"""
defines the `utility function` to be optimized by RL agents
objective function in approaches 1, 2
"""

# imports
import numpy as np

# mapping (D, f) -> u
# assumes all scores are [0, 1]-valued
def synthesize(D_score, f_score):

    # PLACEHOLDER - REPLACE WITH YOUR OWN UTILITY FUNCTION
    # I picked a trivial but relatively well-behaved synethesis function
    # I plan to try out a wider variety of utility functions

    return D_score * f_score

"""
alternate ideas for synthesis function
- min
- Cobb-Douglas
- generally every utility function mentioned in Econ 50
- let Jacob do this part ;*
"""

# sets number of samples p -> L(p) taken by utility fn
NUM_SAMPLES = 10

"""
defines the utility function in terms of
- synthesis function
- prompt
- generator (LLM)
- discriminator (GPT-zero equivalent)
- fitness fn
"""

# p -> u
def u(p, L, D, f, num_samples = NUM_SAMPLES, synthesizer = synthesize):
    
    Lp_batch = L.generate_batch(p, num_samples)
    DLp_batch = D.discriminate_batch(Lp_batch)
    fLp_batch = f.evaluate_batch(p, Lp_batch)

    # synthesize D, f scores
    utility_batch = np.vectorize(synthesizer)(DLp_batch, fLp_batch)
    return np.average(utility_batch)

    # sum_utility = 0
    # for i in len(Lp_batch):
    #     Di = DLp_batch[i]
    #     fi = fLp_batch[i]
    #     ui = synthesizer(Di, fi)

    #     # increment score with current utility
    #     sum_utility += ui
