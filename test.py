"""
test.py
combined testing harness for component packages
provides test method for each component package
"""

# component imports
import generators
import discriminators
import fitness_functions
import utility_function

# for reading in reference passages
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

# test harness for generator package
def test_generators():

    # initializes generator
    generator = generators.GPT2.GPT2()

    # defines prompt
    prompt = ""
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
    discriminator = discriminators.BERT.BERT()

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
    fitness_function = fitness_functions.TrivialFitnessFunction.TrivialFitnessFunction()

    # tests fitness function on reference passages
    for filename, passage in passages.items():
        print("\n\t FILENAME: " + filename)
        # print("\t PASSAGE: " + passage)
        # print("\t PASSAGE LENGTH: " + str(len(passage)))
        fitness = fitness_function.evaluate(passage)
        print("\t FITNESS: ", fitness)

    # tests fitness function on batch of passages
    fitnesses = fitness_function.evaluate_batch(list(passages.values()))

# test harness for utility function
def test_utility_function():
    
    # initializes components
    generator = generators.GPT2.GPT2()
    discriminator = discriminators.BERT.BERT()
    fitness_function = fitness_functions.TrivialFitnessFunction.TrivialFitnessFunction()
    u = utility_function.u

    # evaluates utility of test prompt
    prompt = ""
    num_samples = 1
    utility = u(prompt, generator, discriminator, fitness_function, num_samples)
    print("\n\t UTILITY: ", utility)