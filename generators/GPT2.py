"""
GPT2-based generator
subclass of general Generator class
uses GPT2-large from HuggingFace
https://huggingface.co/gpt2
"""

# imports superclass definition
from Generator import Generator

# imports huggingface infrastructure
# from transformers import GPT2Tokenizer, GPT2 LMHeadModel
from transformers import pipeline

class GPT2(Generator):

    # initializes HuggingFace infrastructure
    def __init__(self):
        
        # defines underlying huggingface generation pipeline
        self.generator = pipeline('text-generation', model='gpt2', return_full_text=False)

    # redefines generator methods
    # invokes huggingface pipeline thru self.generator
    def generate(self, prompt):
        passage_unformatted = self.generator(prompt, max_length=Generator.PASSAGE_LENGTH, do_sample=True)
        return passage_unformatted[0]['generated_text']

    def generate_batch(self, prompt, num_responses):
        passages_unformatted = self.generator(prompt, max_length=Generator.PASSAGE_LENGTH, do_sample=True, num_return_sequences=num_responses)
        return [passage_unformatted['generated_text'] for passage_unformatted in passages_unformatted]
