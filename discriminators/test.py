"""
test harness for discriminator class
creates and samples from a BERT-based discriminator
uses predefined passages (human & machine generated)
"""

# os for file pathing
import os

# json for file I/O
import json

# imports discriminator class
from BERT import BERT

# imports json of passages as dict
# WARNING: must be run from base project directory
proj_path = os.getcwd()
assert(proj_path.endswith("cheatGPT"))

# defines path to reference_passages
INPUT_FILE = "reference_passages.json"
INPUT_PATH = os.path.join(proj_path, INPUT_FILE)
passages = json.load(open(INPUT_PATH, 'r'))

# initializes discriminator
discriminator = BERT()

# tests discriminator on reference passages
passage_list = []
for filename, passage in passages.items():
    print("\n\t FILENAME: " + filename)
    # print("\t PASSAGE: " + passage)
    # print("\t PASSAGE LENGTH: " + str(len(passage)))
    classification = discriminator.discriminate(passage)
    print("\t CLASSIFICATION: ", classification)

    # adds passage to list for batch testing
    passage_list.append(passage)

# tests discriminator on batch of passages
classifications = discriminator.discriminate_batch(passage_list)
