# attention_fsm

## Question

Can a model **track the hidden state of a finite-state machine (DFA)** as it
consumes a token stream? Each token is an input symbol that drives a
deterministic transition; the model must report, at every position, which DFA
state the machine is in after consuming the prefix so far. This is a clean
probe of *sequential state tracking* — the computation an attention head
implements when it carries forward a running summary of the past.

## Setup

**Synthetic generator only.** A fixed 3-state, 4-symbol DFA generates labelled
sequences:

- States `S = {0, 1, 2}`.
- Alphabet `Σ = {A, B, C, D}` encoded as token ids `{0, 1, 2, 3}`.
- Transition table `δ[s][t] = next_state`:

  | from \ token | A(0) | B(1) | C(2) | D(3) |
  |--------------|------|------|------|------|
  | **0**        | 0    | 1    | 2    | 1    |
  | **1**        | 1    | 2    | 0    | 2    |
  | **2**        | 2    | 0    | 1    | 0    |

Each sequence starts from a random state, then applies the transition for each
token. Tokens are uniform iid. The label at position `t` is the DFA state
*after* consuming token `t`. The generator is deterministic given a seed;
`evaluate` always uses the **canonical seed 0**.

## Canonical measurement condition

- `NUM_SEQUENCES = 128`
- `SEQ_LEN = 64`
- `BURNIN = 16` — the first 16 positions are excluded from headline accuracy.
  A model that has not yet observed the start state needs a few steps to lock
  on; we score steady-state tracking, not transient warm-up.
- Canonical seed `= 0`.

## Model function signature

```python
def model_fn(tokens: np.ndarray) -> np.ndarray:
    # tokens: int array [num_sequences, seq_len], values 0..3
    # returns: float array [num_sequences, seq_len, 3] of state logits
    ...
```

`evaluate` takes the argmax state at each position as the prediction. Attempts
never build the payload themselves — they hand `evaluate` a `model_fn` and
receive a ready-to-record payload. `task.random_model_fn()` returns a reference
`model_fn` emitting constant (uniform) logits — chance-level tracking — used by
the pipeline smoke test.

## Payload contract

`task.evaluate(model_fn)` returns exactly these keys (consumed verbatim by
`benchmark.score`):

| key | type | semantics |
|-----|------|-----------|
| `version` | int | payload schema version (matches `benchmark.VERSION`) |
| `seq_len` | int | sequence length (64) |
| `num_sequences` | int | batch size (128) |
| `burnin` | int | positions excluded from headline accuracy (16) |
| `dfa_spec` | dict | `num_states`, `alphabet_size`, `transition`, `token_map` (self-describing; `score` does not read it) |
| `per_position_accuracy` | list[float] | length `seq_len`; mean accuracy over sequences at each position |
| `overall_accuracy` | float | mean accuracy over all positions `>= burnin` |
| `random_baseline_accuracy` | float | `1 / num_states` (chance) |
| `per_state_recall` | list[float] | length `num_states`; post-burnin recall per true state |
| `transition_confusion` | list[list[int]] | `num_states x num_states` confusion counts (post-burnin), indexed `[true][pred]` |

## Metrics

Produced by `benchmark.score(payload)`; `version` is the first key.

| metric | direction | meaning |
|--------|-----------|---------|
| `version` | — | highest-version filter for the dashboard |
| `state_tracking_accuracy_canonical` | bigger better | post-burnin overall accuracy |
| `state_tracking_robustness` | bigger better, `[0,1]` | **headline.** Chance-normalised accuracy `(acc - chance) / (1 - chance)`, clamped to `[0,1]`. 0 = chance, 1 = perfect. |
| `lift_over_random` | bigger better | `acc - random_baseline_accuracy` |
| `random_baseline_accuracy` | — | chance reference (`1/3`) |
| `acc_pos_<p>` | bigger better | per-slice accuracy at sampled positions `p` (early/mid/late) |
| `state_recall_<s>` | bigger better | post-burnin recall for true state `s` |
| `min_state_recall` | bigger better | weakest per-state recall — catches collapse onto one state |
| `late_minus_early_accuracy` | bigger better | accuracy(last quarter) − accuracy(first post-burnin quarter); does tracking hold over depth? |

**Headline:** `state_tracking_robustness`.

## Bump procedure

Bump `benchmark.VERSION` (and `version` in the payload) when the DFA spec, the
canonical condition (seq len, batch, burn-in, seed), any metric formula, or any
payload key name/type changes — and update this contract in the same commit.
Adding a new metric or an optional payload key does not require a bump.
