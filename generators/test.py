"""
test harness for generator class
creates and samples from a GPT2-based generator
"""

# imports generator class
from GPT2 import GPT2

# initializes generator
generator = GPT2()

# defines prompt
prompt = ""
print("\n\t PROMPT: " + prompt)

# generates passage
passage = generator.generate(prompt)

# prints passage
print("\n\t PASSAGE: " + passage)

# generates batch of passages
# N_PASSAGES = 5
# passages = generator.generate_batch(prompt, N_PASSAGES)
