# cheatGPT
### Adversarial prompt-generation to test robustness of human/LLM text classifiers

## Requirements
Environment requirements for this project are listed in `requirements.txt`. I'm working in a conda environment named `proj`, which can be created with the following command:
> conda create --name proj --file requirements.txt

## Standards
- [GPT-2](https://openai.com/blog/better-language-models/) (1.5B) as language model
- The default passage length is 1000 tokens (defined in `generators/Generator.py`)
- for computing the expectation term in utility, the default number of samples is 10 (defined in `utility_function.py`)
    - For efficiency, all classes operating on samples (L, D, f) should be able to take in a list of samples and return a list of results via a `batch` method (example: `generators`)
- for components with prompt-specific and prompt-agnostic versions (that do or don't take in prompts), we define a general prompt-specific class and a prompt-agnostic subclass; setting the prompt attribute to `None` (example: `fitness_functions`)

## Structure

Replacable components (L, D, f) defined in folders
- each equipped with test.py for unit tests

[Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) for external code
- ex: [detectGPT](https://detectgpt.ericmitchell.ai/)
- Jacob sees no reason to edit those submodules (if you find one, please lmk!)