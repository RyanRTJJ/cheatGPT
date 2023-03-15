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



# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


def load_base_model():
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    try:
        mask_model.cpu()
    except NameError:
        pass

    base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def load_mask_model():
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    mask_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'
    buffer_size = 1
    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

    return perturbed_texts


def perturb_texts(texts, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size
    if '11b' in mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])[:-1]
    labels = labels.view(-1)[1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean()


# Get the log likelihood of each text under the base_model
def get_ll(text):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()


def get_lls(texts):
    return [get_ll(text) for text in texts]


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, log=False):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


# get average entropy of each token in the text
def get_entropy(text):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


# save the ROC curve for each experiment, given a list of output dictionaries, one for each experiment, using colorblind-friendly colors
def save_roc_curves(experiments):
    # first, clear plt
    plt.clf()

    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}", color=color)
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves ({base_model_name} - {args.mask_filling_model_name})')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{SAVE_FOLDER}/roc_curves.png")


# save the histogram of log likelihoods in two side-by-side plots, one for human and human perturbed, and one for LLM and LLM perturbed
def save_ll_histograms(experiments):
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of LLM/perturbed LLM on left, human/perturbed human on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            plt.hist([r["LLM_ll"] for r in results], alpha=0.5, bins='auto', label='LLM')
            plt.hist([r["perturbed_LLM_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed LLM')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist([r["human_ll"] for r in results], alpha=0.5, bins='auto', label='human')
            plt.hist([r["perturbed_human_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed human')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/ll_histograms_{experiment['name']}.png")
        except:
            pass


# save the histograms of log likelihood ratios in two side-by-side plots, one for human and human perturbed, and one for LLM and LLM perturbed
def save_llr_histograms(experiments):
    # first, clear plt
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of LLM/perturbed LLM on left, human/perturbed human on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            
            # compute the log likelihood ratio for each result
            for r in results:
                r["LLM_llr"] = r["LLM_ll"] - r["perturbed_LLM_ll"]
                r["human_llr"] = r["human_ll"] - r["perturbed_human_ll"]
            
            plt.hist([r["LLM_llr"] for r in results], alpha=0.5, bins='auto', label='LLM')
            plt.hist([r["human_llr"] for r in results], alpha=0.5, bins='auto', label='human')
            plt.xlabel("log likelihood ratio")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/llr_histograms_{experiment['name']}.png")
        except:
            pass

def get_perturbations(span_length=10, n_perturbations=1):
    load_mask_model()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    human_text = data["human"]
    LLM_text = data["LLM"]

    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=args.pct_words_masked)

    p_LLM_text = perturb_fn([x for x in LLM_text for _ in range(n_perturbations)])
    p_human_text = perturb_fn([x for x in human_text for _ in range(n_perturbations)])

    assert len(p_LLM_text) == len(LLM_text) * n_perturbations, f"Expected {len(LLM_text) * n_perturbations} perturbed samples, got {len(p_LLM_text)}"
    assert len(p_human_text) == len(human_text) * n_perturbations, f"Expected {len(human_text) * n_perturbations} perturbed samples, got {len(p_human_text)}"

    for idx in range(len(human_text)):
        results.append({
            "human": human_text[idx],
            "LLM": LLM_text[idx],
            "perturbed_LLM": p_LLM_text[idx * n_perturbations: (idx + 1) * n_perturbations],
            "perturbed_human": p_human_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })
    
    return results

def get_perturbation_results(results):
    load_base_model()

    for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
        p_LLM_ll = get_lls(res["perturbed_LLM"])
        p_human_ll = get_lls(res["perturbed_human"])
        res["human_ll"] = get_ll(res["human"])
        res["LLM_ll"] = get_ll(res["LLM"])
        res["all_perturbed_LLM_ll"] = p_LLM_ll
        res["all_perturbed_human_ll"] = p_human_ll
        res["perturbed_LLM_ll"] = np.mean(p_LLM_ll)
        res["perturbed_human_ll"] = np.mean(p_human_ll)
        res["perturbed_LLM_ll_std"] = np.std(p_LLM_ll) if len(p_LLM_ll) > 1 else 1
        res["perturbed_human_ll_std"] = np.std(p_human_ll) if len(p_human_ll) > 1 else 1

    return results


def run_perturbation_experiment(results, span_length=10, n_perturbations=1):
    # compute diffs with perturbed
    predictions = {'real': [], 'samples': []}
    for res in results:
        if res['perturbed_human_ll_std'] == 0:
            res['perturbed_human_ll_std'] = 1
            print("WARNING: std of perturbed human is 0, setting to 1")
            print(f"Number of unique perturbed human texts: {len(set(res['perturbed_human']))}")
            print(f"human text: {res['human']}")
        if res['perturbed_LLM_ll_std'] == 0:
            res['perturbed_LLM_ll_std'] = 1
            print("WARNING: std of perturbed LLM is 0, setting to 1")
            print(f"Number of unique perturbed LLM texts: {len(set(res['perturbed_LLM']))}")
            print(f"LLM text: {res['LLM']}")
        predictions['real'].append((res['human_ll'] - res['perturbed_human_ll']) / res['perturbed_human_ll_std'])
        predictions['samples'].append((res['LLM_ll'] - res['perturbed_LLM_ll']) / res['perturbed_LLM_ll_std'])

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    name = f'perturbation_{n_perturbations}'
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': name,
        'predictions': predictions,
        'info': {
            'pct_words_masked': args.pct_words_masked,
            'span_length': span_length,
            'n_perturbations': n_perturbations,
            'n_samples': len(results),
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


def run_baseline_threshold_experiment(criterion_fn, name):
    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    for batch in tqdm.tqdm(range(math.ceil(len(data['human']) / batch_size)), desc=f"Computing {name} criterion"):
        human_text = data["human"][batch * batch_size:(batch + 1) * batch_size]
        LLM_text = data["LLM"][batch * batch_size:(batch + 1) * batch_size]

        for idx in range(len(human_text)):
            results.append({
                "human": human_text[idx],
                "human_crit": criterion_fn(human_text[idx]),
                "LLM": LLM_text[idx],
                "LLM_crit": criterion_fn(LLM_text[idx]),
            })

    # compute prediction scores for real/sampled passages
    predictions = {
        'real': [x["human_crit"] for x in results],
        'samples': [x["LLM_crit"] for x in results],
    }

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': f'{name}_threshold',
        'predictions': predictions,
        'info': {
            'n_samples': len(data['human']),
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }

def load_base_model_and_tokenizer(name):
    base_model_kwargs = {}
    if 'gpt-j' in name or 'neox' in name:
        base_model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in name:
        base_model_kwargs.update(dict(revision='float16'))
    base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=cache_dir)

    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if args.dataset in ['pubmed', 'writing']:
        optional_tok_kwargs['padding_side'] = 'left'
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer

def eval_supervised(data, model):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir=cache_dir).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)

    real, fake = data['human'], data['LLM']

    with torch.no_grad():
        # get predictions for real
        real_preds = []
        for batch in tqdm.tqdm(math.ceil(range(len(real) / batch_size)), desc="Evaluating real"):
            batch_real = real[batch * batch_size:(batch + 1) * batch_size]
            batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            real_preds.extend(detector(**batch_real).logits.softmax(-1)[:,0].tolist())
        
        # get predictions for fake
        fake_preds = []
        for batch in tqdm.tqdm(range(math.ceil(len(fake) / batch_size)), desc="Evaluating fake"):
            batch_fake = fake[batch * batch_size:(batch + 1) * batch_size]
            batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            fake_preds.extend(detector(**batch_fake).logits.softmax(-1)[:,0].tolist())

    predictions = {
        'real': real_preds,
        'samples': fake_preds,
    }

    fpr, tpr, roc_auc = get_roc_metrics(real_preds, fake_preds)
    p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
    print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

    # free GPU memory
    del detector
    torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'info': {
            'n_samples': len(real),
        },
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


if __name__ == '__main__':
    DEVICE = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_perturbation_list', type=str, default="50")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--cache_dir', type=str, default="~/.cache")
    parser.add_argument('--perturb', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--score', action='store_true')
    args = parser.parse_args()

    if args.openai_model is None:
        base_model_name = args.base_model_name.replace('/', '_')
    else:
        base_model_name = "openai-" + args.openai_model.replace('/', '_')

    scoring_model_string = args.scoring_model_name.replace('/', '_')
    START_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    ARGS_FOLDER = f"detect/args"
    if not os.path.exists(ARGS_FOLDER):
        os.makedirs(ARGS_FOLDER)
    print(f"Saving args to absolute path: {os.path.abspath(ARGS_FOLDER)}")

    # write args to file
    with open(os.path.join(ARGS_FOLDER, f'{base_model_name}-{scoring_model_string}-{args.dataset}-{START_DATE}-{START_TIME}.json'), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    mask_filling_model_name = args.mask_filling_model_name
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]

    cache_dir = args.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    PERTURBATIONS_FOLDER = f"detect/perturbations/{base_model_name}/{args.dataset}/"
    if not os.path.exists(PERTURBATIONS_FOLDER):
        os.makedirs(PERTURBATIONS_FOLDER)

    SAVE_FOLDER = f'detect/results/{base_model_name}/{scoring_model_string}/{args.dataset}/'
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    print(f'Loading dataset {args.dataset}...')        
    data = {}
    for key in ['LLM', 'human']:
        data[key] = []
        LOAD_FOLDER = f"inputs/{key}/{args.dataset}"
        for filename in os.listdir(LOAD_FOLDER):
            if filename.startswith(base_model_name):
                with open(os.path.join(LOAD_FOLDER, filename), "r") as f:
                    data[key].append(f.read())

    if args.perturb:
        print(f"Loading mask filling model {mask_filling_model_name}...")
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, cache_dir=cache_dir)
        try:
            n_positions = mask_model.config.n_positions
        except AttributeError:
            n_positions = 512

        mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions, cache_dir=cache_dir)
    
        perturbations_list = []
        for n_perturbations in n_perturbation_list:
            perturbations_list.append(get_perturbations(args.span_length, n_perturbations))
        
        with open(os.path.join(PERTURBATIONS_FOLDER, f'perturbations_list.json'), "w") as f:
            json.dump([perturbations_list, n_perturbation_list], f, indent=4)

    print(f'Loading SCORING model {args.scoring_model_name}...')
    baseline_outputs = []
    
    # commented out just in case loading the model takes up too much memory
    # torch.cuda.empty_cache()
    # base_model, base_tokenizer = load_base_model_and_tokenizer(args.scoring_model_name)
    # load_base_model()
    if args.baseline:
        if args.openai_model is None:
            baseline_outputs.append(run_baseline_threshold_experiment(get_ll, "likelihood"))
            rank_criterion = lambda text: -get_rank(text, log=False)
            baseline_outputs.append(run_baseline_threshold_experiment(rank_criterion, "rank"))
            logrank_criterion = lambda text: -get_rank(text, log=True)
            baseline_outputs.append(run_baseline_threshold_experiment(logrank_criterion, "log_rank"))
            entropy_criterion = lambda text: get_entropy(text)
            baseline_outputs.append(run_baseline_threshold_experiment(entropy_criterion, "entropy"))

        baseline_outputs.append(eval_supervised(data, model='roberta-base-openai-detector'))
        baseline_outputs.append(eval_supervised(data, model='roberta-large-openai-detector'))

        if args.openai_model is None:
            # write likelihood threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"likelihood_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[0], f)
                
            # write rank threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"rank_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[1], f)

            # write log rank threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"logrank_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[2], f)

            # write entropy threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"entropy_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[3], f)
        
        # write supervised results to a file
        with open(os.path.join(SAVE_FOLDER, f"roberta-base-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-2], f)
        
        # write supervised results to a file
        with open(os.path.join(SAVE_FOLDER, f"roberta-large-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-1], f)

    outputs = []

    if args.score:
        print(f'Loading perturbations list from {PERTURBATIONS_FOLDER}...')
        with open(os.path.join(PERTURBATIONS_FOLDER, f'perturbations_list.json'), "r") as f:
            perturbations_list, n_perturbation_list = json.load(f)

        # run perturbation experiments
        for perturbations, n_perturbations in zip(perturbations_list, n_perturbation_list):
            perturbation_results = get_perturbation_results(perturbations)

            output = run_perturbation_experiment(
                perturbation_results, span_length=args.span_length, n_perturbations=n_perturbations)
            outputs.append(output)
        
        with open(os.path.join(SAVE_FOLDER, f"perturbation_experiment_results.json"), "w") as f:
            json.dump(outputs, f, indent=4)

        outputs += baseline_outputs

        save_roc_curves(outputs)
        save_ll_histograms(outputs)
        save_llr_histograms(outputs)