"""
defines synthesis fns (D, f) -> u
for use in utility functions
exports several custom-defined synthesis fns
exports a default synthesis fn
"""

# simple (product) synthesis function
def s_p(D_score, f_score):
    return D_score * f_score

# min synthesis function
def s_min(D_score, f_score):
    return min(D_score, f_score)

# exports default
DEFAULT_SYNTHESIS_FN = s_p