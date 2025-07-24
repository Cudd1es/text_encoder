import numpy as np
import torch
from encoder import TextEncoder, TextSparseEncoder, read_input


class HybridScorer:
    def __init__(self, alpha=0.5):
        """
        :param alpha: weight for dense encoder score is（0~1），weight for sparse encoder score is (1 - alpha)
        """
        self.alpha = alpha

    def cosine_score(self, query_vec, doc_vec):
        # Compute cosine similarity (dense vectors)
        return np.dot(query_vec, doc_vec)

    def sparse_dot_score(self, sparse_q, sparse_d):
        # Sparse vector dot product (default format is torch.sparse.Tensor）
        # They must be converted to dense vectors and then dot products
        q_dense = sparse_q.to_dense()
        d_dense = sparse_d.to_dense()
        return torch.dot(q_dense, d_dense).item()

    def hybrid_score(self, dense_q, dense_d, sparse_q, sparse_d):
        dense_score = self.cosine_score(dense_q, dense_d)
        sparse_score = self.sparse_dot_score(sparse_q, sparse_d)
        return self.alpha * dense_score + (1 - self.alpha) * sparse_score

def rerank(query, documents, top_n=5, alpha=0.5):
    dense_encoder = TextEncoder()
    sparse_encoder = TextSparseEncoder()

    doc_dense_vecs = dense_encoder.encode(documents)
    doc_sparse_vecs = sparse_encoder.encode(documents)

    query_dense_vec = dense_encoder.encode(query)[0]
    query_sparse_vec = sparse_encoder.encode(query)[0]

    scorer = HybridScorer(alpha=alpha)

    results = []
    for i, doc in enumerate(documents):
        dense_score = scorer.cosine_score(query_dense_vec, doc_dense_vecs[i])
        sparse_score = scorer.sparse_dot_score(query_sparse_vec, doc_sparse_vecs[i])
        hybrid = scorer.hybrid_score(query_dense_vec, doc_dense_vecs[i], query_sparse_vec, doc_sparse_vecs[i])
        results.append((doc, hybrid, dense_score, sparse_score))

    # Sort: By hybrid score descending
    results.sort(key=lambda x: x[1], reverse=True)

    # output top N
    print(f"\nTop {top_n} Results for Query: \"{query}\"\n")
    for i, (doc, hybrid, dense, sparse) in enumerate(results[:top_n]):
        print(f"{i+1}. [Score: {hybrid:.4f}]")
        print(f"    Dense: {dense:.4f}, Sparse: {sparse:.4f}")
        print(f"    → {doc}\n")
