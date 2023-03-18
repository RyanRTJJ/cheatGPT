# Jacob's Small and Urgent TODO
- [ ] theoretical foundation for PrGen
    - [ ] superlcass w handler methods (get_ll) for optionality?
    - [ ] write up methods for paper
- [ ] define adversary classes
    - [ ] random search
    - [ ] RLHF
    - [ ] prompt-specific vs agnostic?!
- [ ] robust prompt-specific vs agnostic framework
    - [ ] utility fns
        - implement class structure
    - [ ] adversaries
        - given a prompt-s/a utility fn

# Big TODO
1. set up prebuilt LLM components
    - [x] generator: [GPT-2](https://openai.com/blog/better-language-models/) (1.5B) as language model (specifically [HuggingFace](https://huggingface.co/gpt2))
    - discriminators
        - [x] [detectGPT](https://github.com/eric-mitchell/detect-gpt)
        - [x] [OpenAI](https://huggingface.co/roberta-base-openai-detector)
        - [ ] [GROVER](https://blog.allenai.org/counteracting-neural-disinformation-with-grover-6cf6690d463b)
2. Define domain of prompts p
- [x] Jacob thinks this should be in terms of a fitness fn
    - also need a length constraint? infinite search space seems unstructured
- Ryan thinks this should be a hypothesis class of valid prompts
3. Implement utility function
- [x] Jacob thinks this is in terms of $u = (D, f)(Lp)$
- [x] Interpretable HUD
4. Define fitness fns
- [x] trivial
- [x] as described in milestone report (`fitness_functions/PrGen.py`)
- [ ] other things?
    - [ ] MAUVE (JACOB: MAUVE seems bas)
    - [ ] n-gram
5. Implement search over prompts
- [x] Try Things
- [ ] Random search?
- [ ] Enumerate over known prompts
- [ ] RLHF
6. Define adversarial agent A
- [ ] "adversary" class
    - train method to optimize current promp
    - generate_prompt gives current best prompt
    - looklike on/offline training for RL
7. pipelining of UPT
- [x] stress-testing on more data (Raghav)
- [x] defining a statistically rigorous test (Ryan)
<!--
no time to do these 
- [ ] converting to Discriminator class (not Jacob)
- [ ] testing using existing framework (Jacob)
-->
8. Define more generators
- [ ] GPT-3
    - generation
    - Pr[generation] (for FF)
    - detection
<!-- no time
    - [ ] other stuff?
    - probably needs to be via API
    - how to integrate w fitness fn? -->
9. Remote compute
- [x] get remote server working
- [ ] write gpu-efficient code
- [ ] wrangle gpu from ryan >:()
10. Experiments
- [ ] can we break dumb discriminators (OpenAI?)
- [ ] can we break smart discriminators (DetectGPT, UPT)
- [ ] do discriminators generalize to more expressive LMs?
    - lots of testing w DetectGPT + UPT
    - lots of OpenAI tokens :/

## Potential Codebase Improvements
- [ ] define prompt, passage class
    - standardize in terms of tokenized arrays?
- [ ] be more careful handling/passing around LMs
    - define LM class of HF models?