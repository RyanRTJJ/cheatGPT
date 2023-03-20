"""
trivial fitness function
assigns fitness 0 to any passage
prompt-agnostic (trivially...)
used for testing purposes
"""

# imports superclass definition
from .PAFitnessFunction import PAFitnessFunction

class TrivialFitnessFunction(PAFitnessFunction):

    # fitness of any passage is 0
    def PA_evaluate(self, passage):
        return 0