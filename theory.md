## Notation
- certain components can choose whether to take in a prompt:
    - *prompt-agnostic*: **doesn't** take in a prompt
    - *prompt-specific*: **does** take in a prompt
- component functions are represented by letters
    - L(p): language model
    - D(Lp): discriminator
    - f(p, LAp): fitness function
    - u(D, L): utility function
    - A(p): adversarial agent
- function concatenation omits parenthesis for simplicity
    - ex: LAp = L(A(p))
    - "the language model applied to the adversarial modification of the original prompt"

# 2 approaches

### 1. black-box prompt search
- fitness fn is prompt-agnostic
- hypothesis class of prompts is all valid strings of (English?) tokens with length $\leq n$

### 2. white-box prompt modification
- leverages linguistic structure of prompts?
    - RL framework
        - state is prompt
        - actions are 'atomic' prompt modifications
        - utility is analogous to (1) above
- fitness fn (and thus utility) is NO LONGER prompt-agnostic
    - utility is now prompt-specific

## Open questions/debates
- does utility factor through L?
    - PRO: Jacob
        - college admissions officer example
        - distinction between optimization problem vs optimizing model
    - ANTI: 
        - ease of coding
        - prompt-specific fitness function redundant with prompt modification structure