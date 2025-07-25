# main.py

import argparse
import sys
from encoder import TextEncoder, TextSparseEncoder, read_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", type=str, help="Path to input text or raw text")
    parser.add_argument("--mode", "-m", type=str, choices=["dense", "sparse"], default="dense", help="Encoder type")
    args = parser.parse_args()

    if not args.text:
        parser.print_help()
        sys.exit(1)

    texts = read_input(args.text)

    if args.mode == "dense":
        encoder = TextEncoder()
    elif args.mode == "sparse":
        encoder = TextSparseEncoder()
    else:
        raise ValueError("Unsupported mode.")

    embeddings = encoder.encode(texts)

    if args.mode == "dense":
        print("Dense embedding (first 5 dims):", embeddings[0][:5])
    else:
        print(embeddings)
