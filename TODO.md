Jacob's transcription of Ryan's todo list (from 2/20 whiteboarding sesh)
1. set up prebuilt LLM components
- generator: GPT-2
- discriminators
    - detectGPT
    - OpenAI
    - GROVER
2. Define domain of prompts p
- Jacob thinks this should be in terms of a fitness fn
    - also need a length constraint? infinite search space seems unstructured
- Ryan thinks this should be a hypothesis class of valid prompts
3. Implement utility function
- Jacob thinks this is in terms of $u = (D, f)(Lp)$
4. Define adversarial agent A
- Jacob doesn't wanna do this part