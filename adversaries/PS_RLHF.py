"""
prompt-specific RLHF
optimizes adversary function p -> A(p)
uses trlx to fine-tune a seq2seq model to produce prompts
to optimize utility via RLHF
"""

# imports
from adversaries.PDAd import PDAd
from TIH import trlx

class PS_RLHF(PDAd):

    # initializes 
    def __init__(self, utility_function, p):
        super().__init__(utility_function, p)

    # defines train, test in terms of p_0
    # def PA_train(self):
    #     return self.train(self.p_0)
    
    def PA_test(self):
        return self.test(self.p)