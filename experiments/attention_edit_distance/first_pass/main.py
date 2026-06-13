import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# Load GPT-2 small once
_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
_model = GPT2Model.from_pretrained("gpt2", output_attentions=True).to(DEVICE)
_model.eval()

# Canonical layer/head from the task spec
LAYER = 5
HEAD = 3

@torch.inference_mode()
def model_fn(tokens: np.ndarray) -> np.ndarray:
    """
    Extract attention weights for layer 5, head 3 from GPT-2.
    
    Args:
        tokens: [batch, seq_len] int32 token IDs
        
    Returns:
        [batch, seq_len, seq_len] float32 attention weights
    """
    batch_size, seq_len = tokens.shape
    
    # Convert to torch tensor on GPU
    input_ids = torch.as_tensor(tokens, dtype=torch.long, device=DEVICE)
    
    # Forward pass with attention output
    outputs = _model(input_ids=input_ids)
    attentions = outputs.attentions  # tuple of 12 layers, each [batch, heads, seq_len, seq_len]
    
    # Extract canonical layer and head
    attn = attentions[LAYER][:, HEAD, :, :]  # [batch, seq_len, seq_len]
    
    # Return as numpy on CPU
    return attn.detach().cpu().numpy().astype(np.float32)


def main():
    task = load_task(__file__)
    
    # Run evaluation
    payload = task.evaluate(model_fn)
    
    # Save results
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    
    # Print summary
    print(f"Run saved to: {run_dir}")
    print(f"Model: {payload['model_name']}, Layer: {payload['layer']}, Head: {payload['head']}")
    print("Sweep results:")
    for entry in payload["sweep"]:
        print(f"  Edit distance {entry['edit_distance']}: "
              f"attn_dist = {entry['attn_distance_mean']:.4f} ± {entry['attn_distance_std']:.4f} "
              f"(n={entry['n_pairs']})")


if __name__ == "__main__":
    main()