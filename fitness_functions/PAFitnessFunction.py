"""
prompt-agnostic fitness function
subclass of FitnessFunction
provides prompt-less evaluate and evaluate_batch methods
"""

class PAFitnessFunction(FitnessFunction):

    # evaluates a single passage
    def evaluate(self, passage):
        raise NotImplementedError

    # evaluates a batch of passages
    # default implementation uses python's native vectorization thru list comprehension
    def evaluate_batch(self, passages):
        return [self.evaluate(passage) for passage in passages]
    
    # defaults prompt-specific mehtods to prompt-agnostic ones
    # overrides definitions in FitnessFunction
    def evaluate(self, prompt, passage):
        return self.evaluate(passage)
    
    def evaluate_batch(self, prompt, passages):
        return self.evaluate_batch(passages)