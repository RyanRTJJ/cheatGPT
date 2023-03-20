"""
BERT-based discriminator
subclass of general Discriminator class
uses OpenAI's BERT classifier finetune from HuggingFace
https://huggingface.co/roberta-base-openai-detector
"""

# imports superclass definition
from .Discriminator import Discriminator

# imports huggingface infrastructure
from transformers import pipeline 

import numpy as np

class BERT(Discriminator):

    # MAX PASSAGE LENGTH IS 512 TOKENS
    # see https://huggingface.co/roberta-base-openai-detector
    # or https://huggingface.co/docs/transformers/model_doc/longformer for longer seq.s

    # initializes HuggingFace infrastructure
    def __init__(self):
        
        # defines underlying huggingface classification pipeline
        self.classifier = pipeline('sentiment-analysis',
                                   model='roberta-base-openai-detector',
                                   return_all_scores=True)

    # redefines discriminator methods
    # invokes huggingface pipeline thru self.classifier

    def discriminate(self, passage):
        # gets classification
        classification = self.classifier(passage)

        # returns log Pr[human-generated]
        pr = classification[0][1]['score']
        return np.log(pr)
    
    def discriminate_batch(self, passages):
        # gets classifications
        classifications = self.classifier(passages)

        # returns log Pr[human-generated]
        prs = [classification[1]['score'] for classification in classifications]
        return np.log(prs)