"""
test harness for discriminator class
creates and samples from a BERT-based discriminator
uses predefined passages (human & machine generated)
"""

# os for file pathing
import os

# imports discriminator class
from BERT import BERT

# initializes discriminator
discriminator = BERT()

# defines passages
# reads in each passage from refence_passages
# stores each passage in a dict: name -> passage

# WARNING: must be run from within project directory
# run_path = os.path.dirname(os.path.realpath(__file__))
run_path = os.getcwd()

# truncates run_path to cheatGPT directory
proj_path = run_path[:run_path.index("cheatGPT") + len("cheatGPT")]
assert(proj_path.endswith("cheatGPT"))

# defines path to reference_passages
target_folder = "reference_passages"
target_path = os.path.join(proj_path, target_folder)

# reads in files from target_path
passages = {}
with os.scandir(target_path) as it:
    for entry in it:
        if not entry.name.startswith('.') and entry.is_file():
            # reads in passage
            with open(entry.path, 'r') as file:
                passage = file.read()
                passages[entry.name] = passage