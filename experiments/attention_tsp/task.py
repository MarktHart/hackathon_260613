"""
Task for attention_tsp.

The goal asks whether an attention mechanism can implement the nearest-neighbour
(NN) TSP routing heuristic: at each decode step, attend to the nearest unvisited
city. The setup is fully synthetic — random Euclidean TSP instances — so any
attempt can reproduce it without a downloaded checkpoint.

Exports:
    generate(seed) -> Batch          deterministic instances
    evaluate(model_fn) -> dict        runs model_fn, returns benchmark payload
    random_model_fn() -> ModelFn      random-logit model for the smoke test

Pure Python / NumPy. No I/O, no network.
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np

# --- model_fn contract (see README.md) ---------------------------------------
# A model_fn implements one decode step of a nearest-neighbour TSP router.
# It receives the full city layout, the index of the current city, and a
# boolean mask of already-visited cities, and returns one attention logit per
# city. The evaluator masks visited cities and takes the argmax to choose the
# next city, so a perfect mechanism puts its maximum logit on the nearest
# unvisited city.
#
#   model_fn(coords: np.ndarray (n, 2),
#            current_idx: int,
#            visited: np.ndarray (n,) bool) -> np.ndarray (n,)
ModelFn = Callable[[np.ndarray, int, np.ndarray], np.ndarray]

# --- Canonical measurement condition (see README.md) -------------------------
N_CITIES_SWEEP = [5, 10, 20, 40]   # problem-size axis of the sweep
CANONICAL_N = 10                   # headline / canonical slice
N_INSTANCES = 20                   # random instances per problem size
EVAL_SEED = 42                     # seed evaluate() always generates with
START_CITY = 0                     # tours always start here


@dataclass(frozen=True)
class Batch:
    """A set of TSP instances grouped by problem size.

    coords_list[i] is an (n_i, 2) float32 array of city coordinates in the unit
    square; ns[i] == n_i is its city count.
    """
    coords_list: list
    ns: list


def generate(seed: int = 0) -> Batch:
    """Deterministic batch: N_INSTANCES random instances for each problem size.

    Same seed -> identical coordinates. `seed` shifts the per-instance RNG so
    the whole batch is reproducible.
    """
    coords_list = []
    ns = []

    for ci, n in enumerate(N_CITIES_SWEEP):
        for s in range(N_INSTANCES):
            seed_i = (np.uint64(seed) * np.uint64(1_000_003)
                      + np.uint64(ci) * np.uint64(9_973)
                      + np.uint64(s)) & np.uint64(0xFFFFFFFF)
            r = np.random.default_rng(int(seed_i))
            coords = r.random(size=(n, 2)).astype(np.float32)
            coords_list.append(coords)
            ns.append(int(n))

    return Batch(coords_list=coords_list, ns=ns)


def _dist_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff.astype(np.float64) ** 2, axis=-1))


def _tour_length(dist: np.ndarray, order) -> float:
    total = 0.0
    for a, b in zip(order[:-1], order[1:]):
        total += float(dist[a, b])
    return total


def _nn_tour(dist: np.ndarray, start: int) -> list:
    """The true nearest-neighbour greedy tour — the mechanism's target."""
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    order = [start]
    current = start
    for _ in range(n - 1):
        masked = np.where(visited, np.inf, dist[current])
        nxt = int(np.argmin(masked))
        order.append(nxt)
        visited[nxt] = True
        current = nxt
    return order


def _random_tour(dist: np.ndarray, start: int, rng: np.random.Generator) -> list:
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    order = [start]
    current = start
    for _ in range(n - 1):
        choices = np.flatnonzero(~visited)
        nxt = int(rng.choice(choices))
        order.append(nxt)
        visited[nxt] = True
        current = nxt
    return order


def _safe_ratio(numer: float, denom: float) -> float:
    """nn_len / other_len, clipped to [0, 1]. 0 when other_len is non-positive."""
    if denom <= 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, numer / denom)))


def evaluate(model_fn: ModelFn) -> dict:
    """Greedily decode a tour with `model_fn` for every instance and return a
    payload matching benchmark.score's contract exactly.
    """
    batch = generate(seed=EVAL_SEED)

    by_n = {n: [] for n in N_CITIES_SWEEP}
    base_by_n = {n: [] for n in N_CITIES_SWEEP}

    base_rng = np.random.default_rng(EVAL_SEED)

    for coords, n in zip(batch.coords_list, batch.ns):
        dist = _dist_matrix(coords)

        # Target nearest-neighbour heuristic tour length.
        nn_order = _nn_tour(dist, START_CITY)
        nn_len = _tour_length(dist, nn_order)

        # --- Attempt model: greedy decode, scoring step-wise NN accuracy ---
        visited = np.zeros(n, dtype=bool)
        visited[START_CITY] = True
        current = START_CITY
        order = [START_CITY]
        matches = 0
        steps = 0
        for _ in range(n - 1):
            logits = np.asarray(
                model_fn(coords, int(current), visited.copy()), dtype=np.float64
            ).reshape(-1)
            if logits.shape != (n,):
                raise ValueError(
                    f"model_fn returned shape {logits.shape}, expected ({n},)"
                )
            masked = np.where(visited, -np.inf, logits)
            choice = int(np.argmax(masked))

            true_nn = int(np.argmin(np.where(visited, np.inf, dist[current])))
            matches += int(choice == true_nn)
            steps += 1

            order.append(choice)
            visited[choice] = True
            current = choice

        model_len = _tour_length(dist, order)

        by_n[n].append({
            "nn_accuracy": (matches / steps) if steps else 0.0,
            "tour_length_ratio": _safe_ratio(nn_len, model_len),
        })

        # --- Random baseline under identical conditions ---
        rnd_order = _random_tour(dist, START_CITY, base_rng)
        rnd_len = _tour_length(dist, rnd_order)
        # Expected per-step random accuracy: 1/|unvisited| averaged over steps.
        if n > 1:
            rnd_acc = float(np.mean([1.0 / k for k in range(n - 1, 0, -1)]))
        else:
            rnd_acc = 0.0
        base_by_n[n].append({
            "nn_accuracy": rnd_acc,
            "tour_length_ratio": _safe_ratio(nn_len, rnd_len),
        })

    sweep = []
    random_baseline = []
    for n in N_CITIES_SWEEP:
        recs = by_n[n]
        sweep.append({
            "n": int(n),
            "nn_accuracy": float(np.mean([r["nn_accuracy"] for r in recs])) if recs else 0.0,
            "tour_length_ratio": float(np.mean([r["tour_length_ratio"] for r in recs])) if recs else 0.0,
            "n_instances": len(recs),
        })
        brecs = base_by_n[n]
        random_baseline.append({
            "n": int(n),
            "nn_accuracy": float(np.mean([r["nn_accuracy"] for r in brecs])) if brecs else 0.0,
            "tour_length_ratio": float(np.mean([r["tour_length_ratio"] for r in brecs])) if brecs else 0.0,
            "n_instances": len(brecs),
        })

    return {
        "version": 1,
        "model_name": "synthetic_attention_tsp",
        "canonical_n": CANONICAL_N,
        "n_cities_sweep": list(N_CITIES_SWEEP),
        "sweep": sweep,
        "random_baseline": random_baseline,
    }


def random_model_fn() -> ModelFn:
    """A model_fn with the real signature whose body emits random logits.

    Pure NumPy; used by the pipeline smoke test.
    """
    rng = np.random.default_rng(0)

    def _random_fn(coords: np.ndarray, current_idx: int, visited: np.ndarray) -> np.ndarray:
        n = np.asarray(coords).shape[0]
        return rng.normal(size=n).astype(np.float32)

    return _random_fn


if __name__ == "__main__":
    payload = evaluate(random_model_fn())
    print("payload keys:", list(payload.keys()))
    print("sweep:", payload["sweep"])
