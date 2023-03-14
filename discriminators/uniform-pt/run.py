import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
import custom_datasets
from multiprocessing.pool import ThreadPool
import time

def load_base_model():
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    if args.openai_model is None:
        base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')

def load_base_model_and_tokenizer(name):
    if args.openai_model is None:
        print(f'Loading BASE model {args.base_model_name}...')
        base_model_kwargs = {}
        if 'gpt-j' in name or 'neox' in name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in name:
            base_model_kwargs.update(dict(revision='float16'))
        base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=cache_dir)
    else:
        base_model = None

    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if args.dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer

def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])

def _openai_sample(p):
    if args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = { "engine": args.openai_model, "max_tokens": 200 }
    if args.do_top_p:
        kwargs['top_p'] = args.top_p
    
    r = openai.Completion.create(prompt=f"{p}", **kwargs)
    return p + r['choices'][0].text

# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(texts, base_tokenizer, gpt2_tokenizer, min_words=300, prompt_tokens=35):
    
    # make copy of texts
    original = list(texts)
    
    # encode each text as a list of token ids
    if args.dataset == 'pubmed' or args.dataset == 'writing':
        texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts] 
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    else:
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
        all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

    # decode the prefixes back into text
    prompts = base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)

    if args.openai_model:
        pool = ThreadPool(args.batch_size)
        decoded = pool.map(_openai_sample, prompts)

    else:
        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < min_words:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            # generate outputs only for prompts that need more words
            mask = [len(d.split()) < min_words for d in decoded]
            masked_encoded = {k: v[mask] for k, v in all_encoded.items()}
            sampling_kwargs = {}
            if args.do_top_p:
                sampling_kwargs['top_p'] = args.top_p
            elif args.do_top_k:
                sampling_kwargs['top_k'] = args.top_k
            min_length = 375 if args.dataset in ['pubmed'] else 450
            outputs = base_model.generate(**masked_encoded, min_length=min_length, max_length=500, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)
            new_decoded = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            j = 0
            for i in range(len(decoded)):
                if mask[i]:
                    decoded[i] = new_decoded[j]
                    j += 1
                    
            tries += 1

    if args.openai_model:
        global API_TOKEN_COUNTER

        # count total number of tokens with gpt2_tokenizer
        total_tokens = sum(len(gpt2_tokenizer.encode(x)) for x in decoded)
        API_TOKEN_COUNTER += total_tokens

    # eliminate the prefix from the output
    decoded = [d[len(p) + 1:] for d, p in zip(decoded, prompts)]
    original = [t[len(p) + 1:] for t, p in zip(original, prompts)]
    
    return original, decoded, prompts

def truncate_to_substring(text, substring, idx_occurrence):
    # truncate everything after the idx_occurrence occurrence of substring
    assert idx_occurrence > 0, 'idx_occurrence must be > 0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]

# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb

def generate_samples(raw_data, base_tokenizer, gpt2_tokenizer, batch_size):
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "human": [],
        "sampled": [],
    }

    for batch in range(len(raw_data) // batch_size):
        print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
        original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
        original_text, sampled_text, prompts = sample_from_model(original_text, base_tokenizer, gpt2_tokenizer, min_words=400 if args.dataset in ['pubmed'] else 300)

        for o, s in zip(original_text, sampled_text):
            if args.dataset == 'pubmed' or args.dataset == 'writing':
                o = o.replace(custom_datasets.SEPARATOR, ' ')
            if args.dataset == 'pubmed':
                s = truncate_to_substring(s, 'Question:', 2)

            o, s = trim_to_shorter_length(o, s)

            # add to the data
            data["human"].append(o)
            data["sampled"].append(s)

    data["prompts"] = prompts

    return data

# strip newlines from each example; replace one or more newlines with a single space
def strip_newlines(text):
    return ' '.join(text.split())

def generate_data(dataset, key, n_samples, base_tokenizer, gpt2_tokenizer, preproc_tokenizer):
    # load data
    if dataset in custom_datasets.DATASETS:
        data = custom_datasets.load(dataset, cache_dir)
    else:
        data = datasets.load_dataset(dataset, split='train', cache_dir=cache_dir)[key]
    
    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the mask model)
    # then generate n_samples samples

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [strip_newlines(x) for x in data]

    # try to keep only examples with > 350 words
    if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 350]
        if len(long_data) > 0:
            data = long_data

    random.seed(0)
    random.shuffle(data)

    data = data[:5_000]

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

    # print stats about remainining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return generate_samples(data[:n_samples], base_tokenizer, gpt2_tokenizer, batch_size=batch_size)

if __name__ == '__main__':
    DEVICE = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_key', type=str, default="document")
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--cache_dir', type=str, default="~/.cache")
    args = parser.parse_args()

    API_TOKEN_COUNTER = 0

    if args.openai_model is not None:
        import openai
        assert args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
        openai.api_key = args.openai_key

    if args.openai_model is None:
        base_model_name = args.base_model_name.replace('/', '_')
    else:
        base_model_name = "openai-" + args.openai_model.replace('/', '_')

    START_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    SAVE_FOLDER = f"inputs/args"
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving args to absolute path: {os.path.abspath(SAVE_FOLDER)}")

    # write args to file
    with open(os.path.join(SAVE_FOLDER, f'{base_model_name}-{args.dataset}-({args.n_samples}-{START_DATE}-{START_TIME}.json'), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    n_samples = args.n_samples
    batch_size = args.batch_size

    cache_dir = args.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

    # generic generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name)

    tokenizer_name = 't5-3b' if args.dataset in ['english', 'german'] else 't5-small'
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=512, cache_dir=cache_dir)
    
    load_base_model()

    print(f'Loading dataset {args.dataset}...')
    data = generate_data(args.dataset, args.dataset_key, n_samples, base_tokenizer, gpt2_tokenizer, preproc_tokenizer)
    print(data.keys())
    
    # save every element of the data dict to a separate file in respective folder
    for key, value in data.items():
        SAVE_FOLDER = f"inputs/{key}"
        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)
        print(f"Saving {key} to absolute path: {os.path.abspath(SAVE_FOLDER)}")
        for i, elem in enumerate(value):
            with open(os.path.join(SAVE_FOLDER, f"{i}.txt"), "w") as f:
                f.write(elem)

    print(f"Used an *estimated* {API_TOKEN_COUNTER} API tokens (may be inaccurate)")