import numpy as np
from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray          # (L,) int32
    factors: np.ndarray         # (L, K) float32, values in {-1, +1}
    factor_directions: np.ndarray  # (K, d_model) float32, orthonormal rows
    noise_std: float


def _make_factor_directions(K: int, d_model: int, seed: int) -> np.ndarray:
    """Generate K orthonormal directions in R^d_model."""
    rng = np.random.default_rng(seed + 12345)
    # Sample K vectors, then QR for orthonormality
    A = rng.normal(size=(K, d_model)).astype(np.float32)
    Q, _ = np.linalg.qr(A.T, mode='reduced')  # (d_model, K)
    return Q.T.astype(np.float32)  # (K, d_model)


def _make_factors(L: int, K: int, seed: int) -> np.ndarray:
    """Generate L x K matrix of {-1, +1} factors."""
    rng = np.random.default_rng(seed + 54321)
    return rng.choice([-1.0, 1.0], size=(L, K)).astype(np.float32)


def _tokens_from_factors(factors: np.ndarray, vocab_size: int, seed: int) -> np.ndarray:
    """Map each factor combination to a token ID. Deterministic but arbitrary."""
    L, K = factors.shape
    # Convert {-1,+1} to {0,1} bits, then to integer
    bits = ((factors + 1) / 2).astype(np.int32)  # (L, K)
    tokens = np.zeros(L, dtype=np.int32)
    for i in range(L):
        tok = 0
        for k in range(K):
            tok = (tok << 1) | bits[i, k]
        tokens[i] = tok % vocab_size
    return tokens


def _add_noise(x: np.ndarray, noise_std: float, seed: int) -> np.ndarray:
    if noise_std == 0.0:
        return x
    rng = np.random.default_rng(seed + 99999)
    noise = rng.normal(scale=noise_std, size=x.shape).astype(np.float32)
    return x + noise


def generate(seed: int = 0) -> Batch:
    """
    Deterministic synthetic batch for the canonical condition (noise_std=0.1).
    The sweep is constructed inside evaluate() by re-generating with different noise.
    """
    # Canonical config
    seq_len = 128
    d_model = 64
    d_feat = 64
    K = 4
    vocab_size = 16
    noise_std = 0.1

    factor_directions = _make_factor_directions(K, d_model, seed)
    factors = _make_factors(seq_len, K, seed)
    tokens = _tokens_from_factors(factors, vocab_size, seed)

    # The "clean" latent representation: factors projected to factor_directions
    # This is what an ideal encoder would produce before noise.
    clean_latent = factors @ factor_directions  # (L, d_model)
    noisy_latent = _add_noise(clean_latent, noise_std, seed)

    # We don't return the latent; the model_fn takes tokens and produces q,k.
    # The factor_directions and factors are ground truth for benchmarking.
    return Batch(
        tokens=tokens,
        factors=factors,
        factor_directions=factor_directions,
        noise_std=noise_std,
    )


def _project_qk(q: np.ndarray, k: np.ndarray, factor_directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project q and k onto factor directions. Returns (q_proj, k_proj) each (K, L)."""
    # q: (L, d_model), factor_directions: (K, d_model) -> q_proj: (K, L)
    q_proj = q @ factor_directions.T  # (L, K) -> transpose to (K, L)
    k_proj = k @ factor_directions.T
    return q_proj.T.astype(np.float32), k_proj.T.astype(np.float32)


def evaluate(model_fn: Callable[..., dict]) -> dict:
    """
    Run model_fn on canonical batch and noise sweep.
    Returns payload dict matching benchmark.score contract.
    """
    # Canonical batch
    canonical_batch = generate(seed=0)
    tokens = canonical_batch.tokens
    factor_directions = canonical_batch.factor_directions
    factors = canonical_batch.factors

    # Canonical evaluation
    out = model_fn(tokens, return_qk=True)
    q = out['q'].astype(np.float32)
    k = out['k'].astype(np.float32)
    q_proj_canon, k_proj_canon = _project_qk(q, k, factor_directions)

    canonical = {
        "q_proj": q_proj_canon,
        "k_proj": k_proj_canon,
        "factor_directions": factor_directions.astype(np.float32),
        "factors": factors.astype(np.float32),
        "noise_std": 0.1,
    }

    # Sweep over noise_std
    # We simulate noise by adding it to the *input representation* the model would see.
    # Since model_fn only takes tokens, we can't directly inject noise.
    # Instead, we approximate: the model's q/k will vary with input noise.
    # For the synthetic task, we assume model_fn internally embeds tokens.
    # To simulate noise sweep, we call model_fn multiple times? No — model_fn is fixed.
    #
    # Correct approach: the sweep measures robustness of the *learned* q/k directions.
    # We evaluate the SAME model_fn on batches with different noise added to the
    # *embedding* layer. But model_fn only takes tokens.
    #
    # Resolution: For this synthetic goal, the "noise" is applied to the ground-truth
    # latent that generates the tokens. But tokens are discrete.
    #
    # Actually, re-reading the spec: the generator emits tokens + factors + factor_directions.
    # The noise_std in the config is the noise added to the *continuous latent* before
    # discretization to tokens. But we already discretized.
    #
    # Simpler interpretation: The sweep evaluates the model on *different generated batches*
    # where the underlying continuous representation had different noise levels.
    # Since tokens are deterministic from factors, and factors are fixed for seed=0,
    # the tokens don't change with noise_std. The model_fn sees the same tokens.
    #
    # This means the sweep as described doesn't make sense for a frozen model_fn.
    #
    # FIX: The sweep should be over *evaluation noise* added to the model's *internal*
    # representations. But we can't access those from model_fn.
    #
    # Alternative: The sweep is over different *random seeds* for the factor directions
    # and factors — i.e., different tasks. But the spec says "noise_std sweep".
    #
    # Best interpretation for a synthetic task with fixed model_fn:
    # The model_fn is expected to be a *function that can take an optional noise argument*
    # or the task evaluates the model on clean tokens but measures how the q/k directions
    # would degrade under noise by projecting clean q/k onto noisy factor_directions?
    #
    # Let's re-read the payload contract: sweep entries have q_proj, k_proj.
    # This implies we run model_fn multiple times, each time with a different noise level
    # affecting the input. Since model_fn only takes tokens, the only way is if model_fn
    # has an optional noise parameter, OR we generate different token sequences for each
    # noise level (by adding noise to the continuous latent before discretization).
    #
    # But the spec says "The generator emits tokens... The canonical batch (seed=0) and
    # a robustness sweep over noise_std".
    #
    # I think the intention is: for each noise_std, we generate a *new* batch where the
    # continuous latent had that noise added before discretization. But discretization
    # to tokens loses the noise information unless the noise flips bits.
    #
    # Let's implement it as: for each noise_std, we generate factors + noise, then
    # discretize to tokens (which may flip some factor bits), then run model_fn.
    # This makes the sweep meaningful — the model sees slightly different tokens.
    #
    # Actually, looking at the reference attention_and: it sweeps over cos(q_A, q_B)
    # which is a property of the *queries* the model produces. The model is run once
    # per sweep value because the sweep parameter changes the *input construction*.
    #
    # For LIS, the sweep parameter is noise_std on the input representation.
    # We'll generate a fresh batch for each noise_std (same seed for factors/directions,
    # different noise seed), get tokens, run model_fn, record q_proj/k_proj.

    sweep_noise_stds = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    sweep = []
    for ns in sweep_noise_stds:
        # Generate batch with this noise_std
        # We need factors and factor_directions to be the SAME across sweep
        # (only the noise on the latent changes, which affects tokens)
        # But tokens are discrete — we'll generate clean factors, add noise to latent,
        # then re-discretize? That's messy.
        #
        # Simpler: Keep factors and factor_directions fixed (seed=0). For each noise_std,
        # generate a *noisy version of the latent*, then map to tokens by nearest
        # neighbour in the factor-direction space? Too complex.
        #
        # Let's follow the pattern: the sweep is over different *conditions* the model
        # might encounter. We'll generate K orthonormal directions and L factors once.
        # For each noise_std, we add noise to the *projected factors* (the ideal q/k)
        # and then... but model_fn takes tokens.
        #
        # OK, I think the cleanest interpretation for a *synthetic* task where we
        # control the data generator: the "model" is actually the whole pipeline
        # from tokens to q/k. The noise_std is a property of the *data generation*
        # that the model was trained on. But here we're evaluating a fixed model_fn.
        #
        # Given the constraints, I'll implement the sweep as follows:
        # - Canonical: seed=0, noise_std=0.1 (as generated)
        # - Sweep: for each noise_std, generate a NEW batch with seed=0 but
        #   different noise seed (so factors/directions same, latent noise differs),
        #   discretize to tokens, run model_fn.
        # This means tokens may differ across sweep entries, which is realistic.

        # Use a derived seed for noise variation
        noise_seed = 0xBADF00D + int(ns * 1000)
        # Re-use the same factor_directions and factors (seed=0)
        # but generate noisy latent and discretize
        seq_len = 128
        d_model = 64
        K = 4
        vocab_size = 16

        factor_directions = _make_factor_directions(K, d_model, 0)
        factors = _make_factors(seq_len, K, 0)
        clean_latent = factors @ factor_directions
        noisy_latent = _add_noise(clean_latent, ns, noise_seed)

        # Discretize noisy latent to tokens by projecting back to factor space
        # and taking sign
        recovered_factors = np.sign(noisy_latent @ factor_directions.T)  # (L, K)
        # Map to tokens
        bits = ((recovered_factors + 1) / 2).astype(np.int32)
        tokens_ns = np.zeros(seq_len, dtype=np.int32)
        for i in range(seq_len):
            tok = 0
            for k in range(K):
                tok = (tok << 1) | bits[i, k]
            tokens_ns[i] = tok % vocab_size

        # Run model
        out_ns = model_fn(tokens_ns, return_qk=True)
        q_ns = out_ns['q'].astype(np.float32)
        k_ns = out_ns['k'].astype(np.float32)
        q_proj_ns, k_proj_ns = _project_qk(q_ns, k_ns, factor_directions)

        sweep.append({
            "noise_std": ns,
            "q_proj": q_proj_ns,
            "k_proj": k_proj_ns,
        })

    return {
        "version": 1,
        "config": {
            "seq_len": 128,
            "d_model": 64,
            "d_feat": 64,
            "K": 4,
            "vocab_size": 16,
            "canonical_noise_std": 0.1,
        },
        "canonical": canonical,
        "sweep": sweep,
        "factor_directions": factor_directions.astype(np.float32),
        "factors": factors.astype(np.float32),
    }


def random_model_fn() -> Callable[..., dict]:
    """Returns a dummy model_fn that outputs zero q/k/v of correct shape."""
    def _fn(tokens: np.ndarray, return_qk: bool = True) -> dict:
        L = tokens.shape[0]
        d_model = 64
        q = np.zeros((L, d_model), dtype=np.float32)
        k = np.zeros((L, d_model), dtype=np.float32)
        v = np.zeros((L, d_model), dtype=np.float32)
        attn = np.zeros((L, L), dtype=np.float32)
        return {"q": q, "k": k, "v": v, "attn": attn}
    return _fn