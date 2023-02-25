"""
The Embeddings class from HW 1. This file contains the solution code.
"""
from typing import Iterable

import numpy as np


class Embeddings:
    """
    This class represents a container that holds a collection of words
    and their corresponding word embeddings.
    """

    def __init__(self, words: Iterable[str], vectors: np.ndarray):
        """
        Initializes an Embeddings object directly from a list of words
        and their embeddings.

        :param words: A list of words
        :param vectors: A 2D array of shape (len(words), embedding_size)
            where for each i, vectors[i] is the embedding for words[i]
        """
        self.words = list(words)
        self.indices = {w: i for i, w in enumerate(words)}
        self.vectors = vectors

    def __len__(self):
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.words

    def __getitem__(self, words: Iterable[str]) -> np.ndarray:
        """
        Retrieves embeddings for a list of words.

        :param words: A list of words
        :return: A 2D array of shape (len(words), embedding_size) where
            for each i, the ith row is the embedding for words[i]
        """
        return self.vectors[[self.indices[w] for w in words]]

    @classmethod
    def from_file(cls, filename: str) -> "Embeddings":
        """
        Initializes an Embeddings object from a .txt file containing
        word embeddings in GloVe format.

        :param filename: The name of the file containing the embeddings
        :return: An Embeddings object containing the loaded embeddings
        """
        with open(filename, "r") as f:
            all_lines = [line.strip().split(" ", 1) for line in f]
        words, vecs = zip(*all_lines)
        return cls(words, np.array([np.fromstring(v, sep=" ") for v in vecs]))
