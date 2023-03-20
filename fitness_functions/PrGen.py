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

    # sets device default for gpu parallelization
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LM
    # sends everything to DEVICE by default
    base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    base_model = transformers.AutoModelForCausalLM.from_pretrained('gpt2')
    base_model.to(DEVICE)

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


