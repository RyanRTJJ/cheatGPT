import transformers
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import torch
import torch.nn.functional as F
import re
import os

def print_list_nice(some_list):
    for thing in some_list:
        print("")
        print(thing)

class Perturber:
    """
    Glossary of Terms
    -----------------
    - "span":           A contiguous segment of words to be masked at a time.
    - "buffer_size":    A number of words to put on either end of a span. This is to ensure
                        that you don't mask out contiguous spans.
    - "pct":            A masking percentage
    - "ceil_pct":       A percentage (of how much to mask cap)
    - "int8":           A setting inherited from detect-gpt (unknown)
    - "half":           A setting inherited from detect-gpt (unknown)
    - "cache_dir":      The directory where huggingface stores downloaded models (default: ~/.cache)
    - "mask_top_p":     A float, search "top_p" here for more info: https://huggingface.co/blog/how-to-generate
    - "mask_pattern":   A regex to match all <extra_id_*> tokens, where * is an integer
    - "num_perturbs":   Number of perturbations per story (text).
    """
    SPAN_LENGTH = 2
    BUFFER_SIZE = 1
    PCT = 0.3
    CEIL_PCT = False
    INT8 = False
    HALF = False
    CACHE_DIR = '~/.cache'
    MASK_TOP_P = 1.0
    MASK_PATTERN = re.compile(r"<extra_id_\d+>")
    SEGMENT_LENGTH = 50
    N_PERTURBS = 20

    """
    Different models and what they do
    ---------------------------------
    - MODEL:                The model that assesses perturbation likelihoods
    - MASK_FILLING_MODEL_NAME
    """

    MASK_STRING = '<<<mask>>>'                  # Model-dependent
    MODEL_NAME = 'gpt2'                         # Start with smallest GPT2.
    MASKING_FILLING_MODEL_NAME = 't5-large'     # Is there a reason we don't use same model here?
    BASE_MODEL_NAME = 'gpt2'                    # A 'generic generative model'. Was 'facebook/opt-2.7b' in detect-gpt

    def __init__(self, device='cpu', span_length=None, buffer_size=None):
        self.DEVICE = 'cpu'

        if span_length != None:
            self.SPAN_LENGTH = span_length
        if buffer_size != None:
            self.BUFFER_SIZE = buffer_size

        # Perturbation Evaluating model
        # -----------------------------
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.MODEL_NAME)
        self.model = GPT2Model.from_pretrained(self.MODEL_NAME)

        # Mask filling model
        # ------------------
        int8_kwargs = {}
        half_kwargs = {}
        
        # As of March 2 2023 this is block does nothing; inherited code from detect-gpt
        if self.INT8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif self.HALF:
            half_kwargs = dict(torch_dtype=torch.bfloat16)

        # Perturbing model
        # ----------------
        self.mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self.MASKING_FILLING_MODEL_NAME, 
            **int8_kwargs, 
            **half_kwargs, 
            cache_dir=self.CACHE_DIR)
        try:
            n_positions = self.mask_model.config.n_positions
        except AttributeError:
            n_positions = 512

        self.mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.MASKING_FILLING_MODEL_NAME, 
            model_max_length=n_positions, cache_dir=self.CACHE_DIR)
        
        # Generic Model to score likelihoods
        # ----------------------------------
        print(f'Loading BASE model {self.BASE_MODEL_NAME}...')

        base_model_kwargs = {}
        # This chunk is currently useless, inherited from detect-gpt
        if 'gpt-j' in self.BASE_MODEL_NAME or 'neox' in self.BASE_MODEL_NAME:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in self.BASE_MODEL_NAME:
            base_model_kwargs.update(dict(revision='float16'))
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_NAME, 
            **base_model_kwargs, 
            cache_dir=self.CACHE_DIR)

        optional_tok_kwargs = {}
        # This chunk is currently useless, but we may use facebook/opt-2.7b
        if "facebook/opt-" in self.BASE_MODEL_NAME:
            print("Using non-fast tokenizer for OPT")
            optional_tok_kwargs['fast'] = False
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.BASE_MODEL_NAME, 
            **optional_tok_kwargs, 
            cache_dir=self.CACHE_DIR)
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id

    def sanity_check(self, texts):
        # Check 1: require that all texts have at least 2 full segments in them
        too_short_idxs = []
        for idx, text in enumerate(texts):
            if len(text) < self.SEGMENT_LENGTH * 2:
                too_short_idxs.append(idx)
        
        assert len(too_short_idxs) == 0, f"\nTexts {too_short_idxs} are too short. Ensure each text has >= {2 * self.SEGMENT_LENGTH} words."

    def tokenize_and_mask(self, text):
        """
        @param text:        [string] a piece of text (natural language)

        +---------------+
        | Functionality |
        +---------------+
        Masks out portions of the text by replacing it with <extra_id_NUM>.

        Example
        -------
        Input: "Today was a brilliant day and I did not go out to enjoy the sun."
        Output: "Today was <extra_id_0> and I did not go out to <extra_id_1> the sun."
        """
        tokens = text.split(' ')

        n_spans = self.PCT * len(tokens) / (self.SPAN_LENGTH + self.BUFFER_SIZE * 2)
        if self.CEIL_PCT:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        # Keep masking until you've successfully masked out n_spans number of spans
        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - self.SPAN_LENGTH)
            end = start + self.SPAN_LENGTH
            search_start = max(0, start - self.BUFFER_SIZE)
            search_end = min(len(tokens), end + self.BUFFER_SIZE)
            if self.MASK_STRING not in tokens[search_start:search_end]:
                tokens[start:end] = [self.MASK_STRING]
                n_masks += 1
        
        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == self.MASK_STRING:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text

    def count_masks(self, texts):
        """
        @param texts: List[String] a list of strings

        +---------------+
        | Functionality |
        +---------------+
        Returns a list of integers, where each integer denotes the number of masks counted in
        the text of the same position. The counting is done by simply traversing over each text
        and counting the number of "<extra_id_NUM>" format strings it sees.
        """
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]
    
    def replace_masks(self, masked_texts):
        """
        @param masked_texts: [String] the masked texts (output of self.tokenize_and_mask())

        +---------------+
        | Functionality |
        +---------------+
        replaces each masked span with a sample from T5 mask_model
        TODO: Understand deeply how this works.
        """
        # print(f"\nmasked_texts: {masked_texts}")
        n_expected = self.count_masks(masked_texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        # print(f"\nstop_id: {stop_id}")

        tokens = self.mask_tokenizer(masked_texts, return_tensors="pt", padding=True).to(self.DEVICE)
        # Returns token_ids and attention_mask
        # print(f"\ntokens: {tokens}")

        outputs = self.mask_model.generate(
            **tokens, 
            max_length=150, 
            do_sample=True, 
            top_p=self.MASK_TOP_P, 
            num_return_sequences=1, 
            eos_token_id=stop_id)
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
    
    def extract_fills(self, raw_fills):
        """
        @params raw_fills: List[String] a list of strings that each look like this:
                           "<pad><extra_id_0> was a beautiful<extra_id_1> ... <extra_id_N>"
        
        +---------------+
        | Functionality |
        +---------------+
        Returns a list of lists, where the nested list contains the non-pointy-bracketed parts
        of each text. E.g.:
        [
            ['was a beautiful', 'lived in', ..., 'that it']
        ]
        """
        # remove <pad> from beginning of each text
        raw_fills = [x.replace("<pad>", "").replace("</s>", "").strip() for x in raw_fills]

        # return the text in between each matched mask token
        extracted_fills = [self.MASK_PATTERN.split(x)[1:-1] for x in raw_fills]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills

    def apply_extracted_fills(self, masked_texts, extracted_fills):
        """
        @param masked_texts: List[String] a list of masked texts (output of self.tokenize_and_mask())
        @param extracted_fills: List[String] a list of extracted fills (output of self.extract_fills())

        +---------------+
        | Functionality |
        +---------------+
        Replaces the masked-out parts in masked_texts with the extract fills.
        """
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

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


    def perturb(self, texts):
        """
        @params texts: List[String] A list of texts to perturb (independently)

        +--------+
        | Return |
        +--------+
        A List[String] where each string is a perturbed text.
        """
        print("perturbing...")
        masked_texts = [self.tokenize_and_mask(x) for x in texts]
        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
        
        # detect-gpt has a while loop here to detect "empty" perturbed texts, i.e. some
        # texts become '' post-fill. Don't know why this will happen, but for now, we simply
        # assert that this is not true. Implement fix if necessary.
        assert '' not in perturbed_texts, f"{len([x for x in perturbed_texts if x == ''])} texts have no fill. Please check why."

        return perturbed_texts

    def segment_texts(self, texts):
        """
        @param texts: List[String] list of writing samples

        +--------+
        | Return |
        +--------+
        A List[List[String]] where the nested list of the i-th index contains the i-th 
        writing sample, but chopped up into various segments, depending on self.SEGMENT_LENGTH
        """

        tokenized_texts = [text.split(' ') for text in texts]
        # A list of list of words.

        text_lengths = np.array([len(tokenized_text) for tokenized_text in tokenized_texts])
        n_segments = np.floor(text_lengths / self.SEGMENT_LENGTH).astype(int)
        segment_lengths = np.ceil(text_lengths / n_segments).astype(int)
        segmented_texts = []
        for idx, tokenized_text in enumerate(tokenized_texts):
            text_segments = []
            segment_length = segment_lengths[idx]
            for segment_idx in range(n_segments[idx]):
                text_segment = tokenized_text[segment_idx * segment_length:(segment_idx + 1) * segment_length]
                text_segments.append(" ".join(text_segment))
        
            segmented_texts.append(text_segments)
        
        return segmented_texts

    def perturb_story_n_times(self, segmented_story):
        """
        @param segmented_stories: [List[String] A "story": a list of text segments.
        
        +--------+
        | Return |
        +--------+
        A List[List[String]], which you should think of as a (N_PERTURBS, N_SEGMENTS) grid.
        """
        grid = [self.perturb(segmented_story) for _ in range(self.N_PERTURBS)]
        return grid

    def get_ll_of_grid(self, grid):
        """
        @param grid: List[List[String]], a (N_PERTURBS, N_SEGMENTS) grid.
        """
        ll_grid = np.zeros((len(grid), len(grid[0])))
        with torch.no_grad():
            for set_idx, perturb_set in enumerate(grid):
                print(f"\nGrading set: {set_idx}")
                for segment_idx, perturb_segment in enumerate(perturb_set):
                    print(f"Grading segment_idx: {segment_idx}")
                    tokenized = self.base_tokenizer(perturb_segment, return_tensors="pt").to(self.DEVICE)
                    labels = tokenized.input_ids
                    ll_grid[set_idx, segment_idx] = -self.base_model(**tokenized, labels=labels).loss.item()
        return ll_grid



def perturb_and_score_dir_of_stories(perturber, dir, ll_save_loc, perturbed_txt_save_loc):
    """
    @param perturber: [Perturber] an instance of the Perturber class.
    @param dir: [String] a name of a directory containing .txt files of stories. (One story per file).
    """
    subdirs = os.listdir(dir)
    files = [dir + '/' + subdir for subdir in subdirs if subdir.endswith('.txt')]
    story_names = [subdir[:-4] for subdir in subdirs if subdir.endswith('.txt')]
    print("\nfilenames:")
    print("-" * 30)
    print_list_nice(files)

    texts = []
    for file in files:
        f = open(file, "r")
        text = f.read()
        texts.append(text)
        f.close()
    
    segmented_stories = perturber.segment_texts(texts)
    for idx, segmented_story in enumerate(segmented_stories):
        print(f"\nPROCESSING STORY {idx}...")

        # Save OG story likelihood
        og_story_grid_ll = perturber.get_ll_of_grid([segmented_story])
        np.save(f"{ll_save_loc}/{story_names[idx]}_og.npy", og_story_grid_ll)

        perturb_grid = perturber.perturb_story_n_times(segmented_story) # a (N_PERTURBS, N_SEGMENTS) grid.
        # Save perturbations
        for perturbation_idx, perturbed_segments in enumerate(perturb_grid):
            perturbed_text = ' '.join(perturbed_segments)
            f = open(f'{perturbed_txt_save_loc}/{story_names[idx]}_p_{perturbation_idx}.txt', 'w+')
            f.write(perturbed_text)
            f.close()

        # Save perturbed stories' likelihoods
        perturbation_grid_ll = perturber.get_ll_of_grid(perturb_grid)
        np.save(f"{ll_save_loc}/{story_names[idx]}_pgrid.npy", perturbation_grid_ll)
    
        

perturber = Perturber()

perturb_and_score_dir_of_stories(
    perturber, 
    dir="eda/LLM_stories", 
    ll_save_loc="eda/LLM_results", 
    perturbed_txt_save_loc="eda/LLM_stories_perturbed")

# f = open("short_story.txt", "r")
# texts = [f.read()]
# f.close()


# segmented_stories = perturber.segment_texts(texts)
# for segmented_story in segmented_stories:
#     og_story_grid_ll = perturber.get_ll_of_grid([segmented_story])
#     print(f"\nog_story_grid_ll:")
#     print(og_story_grid_ll)
#     perturb_grid = perturber.perturb_story_n_times(segmented_story)
#     print(f"\nperturb_grid:")
#     print_list_nice(perturb_grid)

#     perturbation_grid_ll = perturber.get_ll_of_grid(perturb_grid)
#     print(f"\ngrid_ll:")
#     print(perturbation_grid_ll)

# # print("")
# # print_list_nice(segmented_texts[0])
# # print(len(segmented_texts[0]))
# # perturber.perturb(segmented_texts)