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

## Encoder

Supports dense and sparse encoder to encoder input text into vectors.

---

## Usage

### Encode text from a file (one line = one input)

```
python main.py -t README.md
```

### Encode text from a CLI string

```
python main.py -t "This is an example sentence."
```

### Use sparse encoder instead of dense

```
python main.py -t "Search-optimized encoding example." -m sparse
```

### Save encoded embeddings to a file

```
python main.py -t input.txt -o output_embeddings.txt
```

- Output file is in plain text format (each row = one embedding).
- Defaults to `embeddings_output` if `--output` is not specified.

---


## Output Format

### Dense Mode (`--mode dense`, default)

- Returns a NumPy array of shape `(num_texts, embedding_dim)`
- Example:
```
[0.021 0.183 0.274 0.095 0.008 ...]
```

### Sparse Mode (`--mode sparse`)

- Returns a PyTorch sparse tensor (`torch.sparse_coo_tensor`)
- Example:
```
tensor(indices=tensor([[0, 0, ..., 0],
                       [1024, 1055, ..., 3098]]),
       values=tensor([0.0183, 0.4408, ..., 0.1129]),
       size=(1, 30522), nnz=92, layout=torch.sparse_coo)
```

> Note: `np.savetxt()` does **not** support saving sparse tensors directly. You'll need to implement custom serialization for sparse output (e.g. convert to dictionary of token weights).

---

### Text Cleaning

Before encoding, input text is cleaned automatically:
* Lowercased
* URLs removed
* Extra whitespace normalized

---


## Hybrid Ranking (Dense + Sparse)

Supports hybrid reranking using both dense and sparse encoders.

---

### Run hybrid search from command line

`ranker_main.py`: Perform hybrid search and rerank documents using both encoders.


```python ranker_main.py --query "your search query" --doc_path documents/document_example.txt```

#### Optional arguments:


* --top_n: Number of top results to return. Default: 5

* --alpha: Weight for dense similarity in hybrid score. Range: 0.0 (only sparse) to 1.0 (only dense). Default: 0.5

* --doc_path: Path to your document list file (one document per line). Default: documents/document_example.txt

#### Example

```python ranker_main.py --query "best vpn for gaming" --top_n 3 --alpha 0.7```

This will:

* Encode query using both dense and sparse encoders

* Rerank the document list by a hybrid score: 0.7*dense_score + 0.3*sparse_score

* Output top 3 most relevant results

---

### Output example

```angular2html
Top 3 Results for Query: "best vpn for gaming"

1. [Score: 1.2345]
    Dense: 0.8765, Sparse: 0.4789
    → ProtonVPN offers high-speed servers ideal for gaming and streaming.

2. [Score: 1.0154]
    Dense: 0.7123, Sparse: 0.6063
    → NordVPN is one of the most popular VPNs with low latency features.

3. [Score: 0.9457]
    Dense: 0.6870, Sparse: 0.5171
    → Surfshark is affordable and offers good performance in Asia.

```

---

### Example Documents File

The --doc_path file should contain one document per line:

```angular2html
ProtonVPN offers high-speed servers ideal for gaming and streaming.
NordVPN is one of the most popular VPNs with low latency features.
Surfshark is affordable and offers good performance in Asia.
...
```


### License

MIT License. Use at your own risk.


