from sentence_transformers import SentenceTransformer
import re
import argparse
import os
import sys

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", type=str, help="path to input text to be encoded")
    args = parser.parse_args()
    if not args.text:
        parser.print_help()
        sys.exit(1)

    texts = read_input(args.text)

    embeddings = TextEncoder().encode(texts)

    # output test
    print("First sentence embedding (first 5 dims):")
    print(embeddings[0][:5])