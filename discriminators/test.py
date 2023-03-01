"""
test harness for discriminator class
creates and samples from a BERT-based discriminator
uses predefined passages (human & machine generated)
"""

# imports discriminator class
from BERT import BERT

# initializes discriminator
discriminator = BERT()

# defines passages
# reads in each passage from refence_passages
# stores each passage in a dict: name -> passage
passages = {}