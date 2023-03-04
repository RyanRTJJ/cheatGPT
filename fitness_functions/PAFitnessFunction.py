"""
prompt-agnostic fitness function
subclass of FitnessFunction
provides prompt-less evaluate and evaluate_batch methods
"""

# imports superclass definition
from .FitnessFunction import FitnessFunction

class PAFitnessFunction(FitnessFunction):

    # defines a prompt-agnostic fitness functions


    # evaluates a single passage
    def PA_evaluate(self, passage):
        raise NotImplementedError

    # evaluates a batch of passages
    # default implementation uses python's native vectorization thru list comprehension
    def PA_evaluate_batch(self, passages):
        return [self.PA_evaluate(passage) for passage in passages]
    
    # defaults prompt-specific mehtods to prompt-agnostic ones
    def evaluate(self, prompt, passage):
        return self.PA_evaluate(passage)
    
    def evaluate_batch(self, prompt, passages):
        return self.PA_evaluate_batch(passages)