#  Text Encoder

A unified text encoder supporting both **dense** and **sparse** vector representations:

- **Dense Encoder**: Uses [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), optimized for general-purpose semantic similarity.
- **Sparse Encoder**: Uses [`opensearch-neural-sparse-encoding-doc-v2-distill`](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill), optimized for search-relevance and token-level interpretability.

---

##  Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```
---

### Usage

Encode text from a file (one line = one text input)

```python encoder.py -t requirements.txt```

Encode text from CLI input

```python encoder.py -t "This is an example sentence."```

Use sparse encoder instead of dense

```python encoder.py -t "Search-optimized encoding example." -m sparse```

To view help:

```python encoder.py --help```

If --text is not provided, help will be printed automatically.

---

### Output Format
Dense Mode (--mode dense, default)

* Returns a NumPy array of shape (num_texts, embedding_dim)
* Example:

```Dense embedding (first 5 dims): [0.021 0.183 0.274 0.095 0.008]```

Sparse Mode (--mode sparse)

* Returns a PyTorch sparse tensor (torch.sparse_coo_tensor)
* Example:

```
tensor(indices=tensor([[0, 0, 0, ..., 0, 0],
                       [1012, 1024, 1055, ..., 2952, 3098]]),
       values=tensor([0.3221, 0.0183, 0.4408, ..., 0.0312, 0.1129]),
       size=(1, 30522), nnz=92, layout=torch.sparse_coo)
```

Each token in the vocabulary is assigned a weight, but only non-zero values are stored.

---

### Text Cleaning

Before encoding, input text is cleaned automatically:
* Lowercased
* URLs removed
* Extra whitespace normalized

---
### License

MIT License. Use at your own risk.


