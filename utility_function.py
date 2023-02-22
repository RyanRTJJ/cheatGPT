"""
defines the `utility function` to be optimized by RL agents
objective function in approaches 1, 2
"""

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

# Lp -> u
def uLp(s, Lp, D, f):
    return s(D(Lp), f(Lp))

# p -> u
def u(s, p, L, D, f):
    
    Lp_ensemble = L.generate_ensemble(p, NUM_SAMPLES)
    uLp_ensemble = [uLp(s, Lp, D, f) for Lp in Lp_ensemble]
    return sum(uLp_ensemble) / len(uLp_ensemble)