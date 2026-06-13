# attention_cyk - first_pass

## What I did
I created a first-pass hand-built model that directly implements an attention-like mechanism. The model function takes a bracket sequence and a span (i,j) and returns scores over possible split points. The mechanism assigns higher weights to positions where the bracket depth reaches zero (balance points) and moderate weights to other potential split points, simulating what an effective CYK attention mechanism should do.

## Why this visualisation
The visualisation shows how the hand-built mechanism identifies correct split points for CYK parsing. It highlights the importance of bracket balance points and demonstrates that a mechanism should assign highest attention scores to these points to effectively implement the CYK algorithm. The simple structure allows us to understand the direct relationship between bracket patterns and attention scores.