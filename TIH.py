"""
trlx import helper
cd's into trlx and imports
"""

# os for file pathing
import os

# WARNING: must be run from base project directory
proj_path = os.getcwd()

# checks for correct proj folder
assert(proj_path.endswith("cheatGPT"))
 
# defines path to trlx dir
target_folder = "trlx"
target_path = os.path.join(proj_path, target_folder)

# cds into target_path
os.chdir(target_path)

# imports trlx
import trlx