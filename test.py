"""
test.py
combined testing harness for component packages
provides test method for each component package
"""

# imports
import pprint

# component imports
import generators
import discriminators
import fitness_functions
import utility_functions
import adversaries

# for reading in reference passages + prompts
import os
import json

# imports json of reference passages as dict
# WARNING: must be run from base project directory
proj_path = os.getcwd()
assert(proj_path.endswith("cheatGPT"))

# defines path to reference_passages
INPUT_FILE = "reference_passages.json"
INPUT_PATH = os.path.join(proj_path, INPUT_FILE)
passages = json.load(open(INPUT_PATH, 'r'))

# defines path to reference_prompts
FILENAME = "jacob.txt"
INPUT_FILE = f"reference_prompts/{FILENAME}"
INPUT_PATH = os.path.join(proj_path, INPUT_FILE)

# reads input file line by line
with open(INPUT_PATH, 'r') as f:
    prompts = f.readlines()

# initializes default components
generator = generators.GPT2()
discriminator = discriminators.BERT()
fitness_function = fitness_functions.PaperPrGen()
utility_function = utility_functions.PFUF(generator, discriminator, fitness_function)
adversary = adversaries.UninformedSearch(utility_function, prompts)

# test harness for generator package
def test_generators():

    # initializes generator
    # generator = generators.GPT2.GPT2()

    # defines prompt
    prompt = "what do you get when you multiply six by nine?"
    print("\n\t PROMPT: " + prompt)

    # generates passage
    passage = generator.generate(prompt)

    # prints passage
    print("\n\t PASSAGE: " + passage)

    # generates batch of passages
    N_PASSAGES = 2
    passages = generator.generate_batch(prompt, N_PASSAGES)

# test harness for discriminator package
def test_discriminators():
    
    # initializes discriminator
    # discriminator = discriminators.BERT.BERT()

    # tests discriminator on reference passages
    for filename, passage in passages.items():
        print("\n\t FILENAME: " + filename)
        # print("\t PASSAGE: " + passage)
        # print("\t PASSAGE LENGTH: " + str(len(passage)))
        classification = discriminator.discriminate(passage)
        print("\t CLASSIFICATION: ", classification)

    # tests discriminator on batch of passages
    classifications = discriminator.discriminate_batch(list(passages.values()))

# test harness for fitness function package
def test_fitness_functions():
    
    # initializes fitness function
    # fitness_function = fitness_functions.PrGen.PrGen()

    # dummy prompt for prompt-specific fitness functions
    null_prompt = ""

    # tests fitness function on reference passages
    for filename, passage in passages.items():
        print("\n\t FILENAME: " + filename)
        # print("\t PASSAGE: " + passage)
        # print("\t PASSAGE LENGTH: " + str(len(passage)))
        fitness = fitness_function.evaluate(null_prompt, passage)
        print("\t FITNESS: ", fitness)

    # tests fitness function on batch of passages
    fitnesses = fitness_function.evaluate_batch(null_prompt, list(passages.values()))

# test harness for utility function
def test_utility_functions():
    
    # initializes components
    # generator = generators.GPT2.GPT2()
    # discriminator = discriminators.BERT.BERT()
    # fitness_function = fitness_functions.TrivialFitnessFunction.TrivialFitnessFunction()
    # utility_function = utility_functions.PFUF.PFUF(generator, discriminator, fitness_function)

    # evaluates utility of test prompt
    prompt = ""
    # num_samples = 1
    ui = utility_function.PA_u_interpretable(prompt)
    
    # pretty prints utility json ui
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(ui)

# test harness for adversaries
def test_adversaries():
    
    # trains & tests adversary
    # gets optimal prompt
    adversary.train()
    optimal_prompt = adversary.test()

    # evaluates returned prompt
    print("\n\t OPTIMAL PROMPT: " + optimal_prompt)
    ui = utility_function.PA_u_interpretable(optimal_prompt)
    
    # pretty prints utility json ui
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(ui)
