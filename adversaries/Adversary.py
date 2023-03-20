"""
defines adversary class
direct access to utility fn
train method to find good prompts
test method for scored prompts
"""

# imports


# defines adversary class
class Adversary:

    # initializes adversary
    def __init__(self, utility_function):
        self.utility_function = utility_function

    # stubs for train, test methods
    def train(self):
        raise NotImplementedError

    def test(self, prompt):
        raise NotImplementedError