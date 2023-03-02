import numpy as np
from enum import Enum

class Distorter:
  actions = {
    'COMMAND_STYLISTIC_MODIFIER': 0,
    'COMMAND_UNEXPECTED_LANGUAGE': 1,
  }

  unexpected_language_commands = {
    'using unusual language,',
    'using surprising language,',
    'using creative license,'
  }
  
  def __init__(self):
    return

distorter = Distorter()
print(distorter.actions['COMMAND_STYLISTIC_MODIFIER'])