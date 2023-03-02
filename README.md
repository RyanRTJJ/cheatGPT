# cheatGPT
### Adversarial prompt-generation to test robustness of human/LLM text classifiers

## How2Use
- Use `python=3.9` (`>= 3.8` is fine), but 3.9 is the conda default for v3.0 and above. 
- If `detect-gpt` folder is empty, just run `git submodule init`, and `git submodule update` while in the `detect-gpt` folder.

## Bulletin
- conda env currently nonfunctional
    - existing env creation works
    - detect-gpt seems to require CUDA, which is incompatible with macOS
    - only submodule so far is detect-gpt, so submodule infrastructure is currently unused

## Requirements
I (Jacob) am working in a conda environment named `proj`, which can be created with the following commands:

### Option 1: environment.yml
> conda env create --file environment.yml

### Option 2: requirements.txt
> conda create --name proj --file requirements.txt

Environment requirements for this project are listed in `environment.yml`. I'm working in a conda environment named `proj`, which can be created with the following command:
> conda env create --file environment.yml

## Standards
- [GPT-2](https://openai.com/blog/better-language-models/) (1.5B) as language model (specifically [HuggingFace](https://huggingface.co/gpt2))
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

To initialize submodules, run the following commands in the cloned repo:

> git submodule init

> git submodule update


# Ryan's Scratchboard
## Playing with DetectGPT
I used this script to run `run.py` (found in `n_perturb.sh`): 

```zsh
python run.py --output_name n_perturb --base_model_name facebook/opt-2.7b --mask_filling_model_name t5-large --n_perturbation_list 1,10,100,1000 --n_samples 100 --pct_words_masked 0.3 --span_length 2 --dataset writing
```

Reference list for `args`:
```python
args = {
  'dataset': 'writing',
  'dataset_key': 'document',
  'pct_words_masked': 0.3,
  'span_length': 2,
  'n_samples': 100,
  'n_perturbation_list': '1,10,100,1000',
  'n_perturbation_rounds': 1,
  'base_model_name': 'facebook/opt-2.7b',
  'scoring_model_name': '',
  'mask_filling_model_name': 't5-large',
  'batch_size': 50,
  'chunk_size': 20,
  'n_similarity_samples': 20,
  'int8': False,
  'half': False,
  'base_half': False,
  'do_top_k': False,
  'top_k': 40,
  'do_top_p': False,
  'top_p': 0.96,
  'output_name': 'n_perturb',
  'openai_model': None,
  'openai_key': None,
  'baselines_only': False,
  'skip_baselines': False,
  'buffer_size': 1,
  'mask_top_p': 1.0,
  'pre_perturb_pct': 0.0,
  'pre_perturb_span_length': 5,
  'random_fills': False,
  'random_fills_tokens': False,
  'cache_dir': '~/.cache'
}
```