from sentence_transformers import SentenceTransformer
from sentence_transformers.sparse_encoder import SparseEncoder
import re
import os


class TextEncoder:
    def __init__(self, model_name="all-MiniLM-L6-v2", normalize=True):
        # normalize it due to the consideration of cosine similarity
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.cleaner = TextCleaner()

    def encode(self, texts):
        """
        Encodes texts into vectors
        :param:
            texts: str or list of str
        :return:
            a numpy array of shape (num_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = [self.cleaner.clean(t) for t in texts]
        output_embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=self.normalize)

        return output_embeddings

class TextSparseEncoder:
    # sparse encoding method
    def __init__(self, model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"):
        device = "cpu"  # force CPU due to MPS sparse op limitation
        self.sparse_encoder = SparseEncoder(model_name, device=device)
        self.cleaner = TextCleaner()

    def encode(self, texts):
        """
        Encodes texts into vectors
        :param:
            texts: str or list of str
        :return:
             a tensor of weight vector: torch.sparse.Tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = [self.cleaner.clean(t) for t in texts]
        output_embeddings = self.sparse_encoder.encode(texts)

        return output_embeddings



class TextCleaner:
    def __init__(self, lowercase=True, remove_urls=True, normalize_space=True):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.normalize_space = normalize_space

    def clean(self, text:str) -> str:
        """
        Cleans text
        :param text: str
        :return: cleaned text
        """
        text = text.strip()

        if self.lowercase:
            text = text.lower()
        if self.remove_urls:
            text = re.sub(r"http\S+", "", text)
        if self.normalize_space:
            text = re.sub(r"\s+", " ", text)

        return text

def read_input(source):
    """
    :param source: input text
    :return: string or list of input text
    """

    if os.path.exists(source):
        with open(source, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    else:
        return [source]
