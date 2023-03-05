import numpy as np
from enum import Enum

class Distorter:
  actions = {
    'COMMAND_STYLISTIC_MODIFIER': 0,
    'COMMAND_UNEXPECTED_LANGUAGE': 1,
  }

  _unexpected_language_commands = {
    'using unusual language,',
    'using surprising language,',
    'using creative license,'
  }
  
  def __init__(self):
    return
  
  def take_action(self, input_prompt):
    """
    @param input_prompt: a string, e.g. "write me a paragraph about how the sky is blue"
    """

distorter = Distorter()
print(distorter.actions['COMMAND_STYLISTIC_MODIFIER'])