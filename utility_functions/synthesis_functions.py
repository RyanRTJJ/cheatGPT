"""
defines synthesis fns (D, f) -> u
for use in utility functions
exports several custom-defined synthesis fns
exports a default synthesis fn

EVERYTHING IS IN LOG SPACE
"""

# imports
import numpy as np

# simple (product) synthesis function
def s_p(D_score, f_score):
    return D_score + f_score

# min synthesis function
def s_min(D_score, f_score):
    return min(D_score, f_score)

# weighted synthesis fn
# weight vector lambda defined below
LAMBDA = np.array([1, 0.1])
def s_w(D_score, f_score):
    return LAMBDA.dot(np.array([D_score, f_score]))

# exports default
DEFAULT_SYNTHESIS_FN = s_w