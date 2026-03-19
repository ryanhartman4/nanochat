"""Read x0_lambdas and resid_lambdas from a saved checkpoint."""
import sys
import torch
from nanochat.gpt import GPT, GPTConfig

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint .pt file")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # Extract scalars directly from state dict
    x0 = state.get("x0_lambdas")
    resid = state.get("resid_lambdas")

    if x0 is None or resid is None:
        print("Could not find x0_lambdas or resid_lambdas in checkpoint")
        print("Keys:", [k for k in state.keys() if "lambda" in k.lower()])
        sys.exit(1)

    print(f"{'Layer':>6}  {'x0_lambda':>12}  {'resid_lambda':>12}")
    print("-" * 34)
    for i in range(len(x0)):
        print(f"{i:>6}  {x0[i].item():>12.6f}  {resid[i].item():>12.6f}")

if __name__ == "__main__":
    main()
