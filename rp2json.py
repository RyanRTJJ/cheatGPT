"""
helper script
compiles textfiles from reference_passages into a single json file
FORMAT: dict of {filename: passage}
"""

# os for file pathing
import os

# json for file I/O
import json

# defines path to reference_passages
# WARNING: must be run from base project directory
proj_path = os.getcwd()

# checks for correct proj folder
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

# outputs passages as json to OUTPUT_PATH
OUTPUT_FILE = "reference_passages.json"
OUTPUT_PATH = os.path.join(proj_path, OUTPUT_FILE)
with open(OUTPUT_PATH, 'w') as file:
    json.dump(passages, file)