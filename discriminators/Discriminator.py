"""
passage -> {human, Generator} classifier
equipped with individual and batch classification methods
prompt-agnostic; only cares about human- or bot-like
"""

class Discriminator:

    # classifies a single passage
    # predicts Pr \in [0, 1] that passage is HUMAN-generated
    # returns log pr
    # to be maximized by adversary
    def discriminate(self, passage):
        raise NotImplementedError

    # classifies a batch of passages
    # default implementation uses python's native vectorization thru list comprehension
    def discriminate_batch(self, passages):
        return [self.discriminate(passage) for passage in passages]
    

