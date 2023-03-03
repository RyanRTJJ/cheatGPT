"""
LLMs responsible for generating text given a prompt
equipped with individual and batch generation methods
"""

class Generator:

    # length of passage generated
    PASSAGE_LENGTH = 500

    # generates PASSAGE_LENGTH many tokens given prompt
    def generate(self, prompt):
        raise NotImplementedError
    
    # generates an ensemble of responses
    # default implementation uses python's native vectorization thru list comprehension
    def generate_batch(self, prompt, num_responses):
        return [self.generate(prompt) for _ in range(num_responses)]