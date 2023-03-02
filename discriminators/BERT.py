"""
BERT-based discriminator
subclass of general Discriminator class
uses OpenAI's BERT classifier finetune from HuggingFace
https://huggingface.co/roberta-base-openai-detector
"""

# imports superclass definition
from Discriminator import Discriminator

# imports huggingface infrastructure
from transformers import pipeline 

class BERT(Discriminator):

    # initializes HuggingFace infrastructure
    def __init__(self):
        
        # defines underlying huggingface classification pipeline
        self.classifier = pipeline('sentiment-analysis', model='roberta-base-openai-detector')

    # redefines discriminator methods
    # invokes huggingface pipeline thru self.classifier
    def discriminate(self, passage):
        classification = self.classifier(passage)
        return classification[0]['score']

    def discriminate_batch(self, passages):
        classifications = self.classifier(passages)
        return [classification[0]['score'] for classification in classifications]