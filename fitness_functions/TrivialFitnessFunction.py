"""
trivial fitness function
assigns fitness 1 to any passage
prompt-agnostic (trivially...)
used for testing purposes
"""

# imports superclass definition
from PAFitnessFunction import PAFitnessFunction

class TrivialFitnessFunction(PAFitnessFunction):

    # fitness of any passage is 1
    def evaluate(self, passage):
        return 1