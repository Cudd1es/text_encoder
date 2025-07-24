import argparse
from ranker import rerank
from encoder import read_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, required=True, help="Input query string")
    parser.add_argument("--doc_path", "-d", type=str, default="documents/document_example.txt", help="Path to document file")
    parser.add_argument("--top_n", "-n", type=int, default=5, help="Number of top results to show")
    parser.add_argument("--alpha", "-a", type=float, default=0.5, help="Weight of dense score in final result (0~1)")
    args = parser.parse_args()

    documents = read_input(args.doc_path)
    rerank(args.query, documents, args.top_n, args.alpha)