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

class BERT(Discriminator):

    # NOT WORKING
    # changed default passage length to 500 tokens as workaround
    # might only be defined on <510 tokens
    # source: https://huggingface.co/roberta-base-openai-detector#how-to-get-started-with-the-model

    # max passage length is 1024 > 1000 tokens
    # default is 512 tokens
    # MAX_PASSAGE_LENGTH = 1024

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

        # returns Pr[human-generated]
        return classification[0][1]['score']
    
    def discriminate_batch(self, passages):
        # gets classifications
        classifications = self.classifier(passages)

        # returns Pr[human-generated]
        return [classification[1]['score'] for classification in classifications]