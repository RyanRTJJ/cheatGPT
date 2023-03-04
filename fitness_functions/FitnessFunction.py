
"""
passage -> fitness score
equipped with individual and batch fitness methods
prompt-specific by default; "grades" a passage by coherence to prompt
"""

class FitnessFunction:

    # evaluates a single passage
    def evaluate(self, prompt, passage):
        raise NotImplementedError

    # evaluates a batch of passages
    # default implementation uses python's native vectorization thru list comprehension
    def evaluate_batch(self, prompt, passages):
        return [self.evaluate(self, prompt, passage) for passage in passages]