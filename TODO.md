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
- Jacob doesn't wanna do this part
- maybe nobody's doing this part?
7. pipelining of UPT
- [ ] stress-testing on more data (Raghav)
- [ ] defining a statistically rigorous test (Ryan)
- [ ] converting to Discriminator class (not Jacob)
- [ ] testing using existing framework (Jacob)
8. Define more generators
- [ ] GPT-3
    - use OpenAI API
    - is Raghav already doing this for pipelining UPT?
- [ ] other stuff?
    - probably needs to be via API
    - how to integrate w fitness fn?
9. Remote compute
- [x] get remote server working
- [ ] write gpu-efficient code
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