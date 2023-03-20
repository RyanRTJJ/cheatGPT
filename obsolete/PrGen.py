"""
'PrGen': Probabiliy of Generation
Computes fitness of passage Lp given prompt p_0 as Pr[Lp | p_0]
compares actual p_0 to null prompt to compute `explanatory power'
"""

# base class
from fitness_functions.FitnessFunction import FitnessFunction

# other imports
import transformers
from transformers import GPT2Tokenizer, GPT2Model
import torch
# import torch.nn.functional as F

import numpy as np
from scipy.special import expit as sigmoid

class PrGen(FitnessFunction):

    # global constants

    # compares given prompt to baseline prompt pb
    # we choose null prompt as modeling decision
    baseline_prompt = ""

    # linear transformation from probability to score
    m = 1
    b = 0

    # converts probability to score
    # y = m * x + b from class constants
    @classmethod
    def LT(cls, x):
        return cls.m * x + cls.b

    # sets device default for gpu parallelization
    DEVICE = 'cpu'

    # LM
    base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    base_model = transformers.AutoModelForCausalLM.from_pretrained('gpt2')

    # base method for computing Pr[Lp | p_0]
    # adapted from DetectGPT
    # https://github.com/eric-mitchell/detect-gpt/blob/main/run.py
    # RETURNS NEG NUMBER = log(Pr) for consistency with detect-gpt
    @classmethod
    def get_ll(cls, text):

        # handles null prompt as a special case
        if not text: return 0

        with torch.no_grad():
            # computes average ll per token
            tokenized = cls.base_tokenizer(text, return_tensors="pt").to(cls.DEVICE)
            labels = tokenized.input_ids
            avg_ll = - cls.base_model(**tokenized, labels=labels).loss.item()

            # scales by number of tokens
            n_tokens = labels.shape[1]
            return avg_ll * n_tokens

    # evaluates a single passage
    def evaluate(self, prompt, passage):

        # computes needed log probabilities
        # returns positive log loss!!!

        # computes Pr[passage | prompt]
        Lp_given_p0 = self.get_ll(prompt + passage) - self.get_ll(prompt)

        # computes Pr[passage | baseline prompt]
        Lp_given_pb = self.get_ll(self.baseline_prompt + passage) - self.get_ll(self.baseline_prompt)

        # computes delta log probability
        # positive DLP <-> p_0 is more explanatory
        delta_log_prob = Lp_given_p0 - Lp_given_pb

        # converts delta probability to score
        return sigmoid(self.LT(delta_log_prob))


