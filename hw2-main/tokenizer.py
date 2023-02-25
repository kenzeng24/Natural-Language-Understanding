"""
Code for Problem 2 of HW 2.
"""
from typing import Dict, List

import torch
from bs4 import BeautifulSoup
#from datasets.arrow_dataset import Batch
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize


class Tokenizer:
    """
    Problems 2d and 2e: Complete the implementation of this class based
    on the docstrings and the usage examples in the problem set.

    This class is a wrapper around nltk's TreebankWordTokenizer. Its
    function is to tokenize a raw mini-batch and turn it into a valid
    input for an LSTMSentimentClassifier.
    """

    def __init__(self, vocab: List[str]):
        self.words = list(vocab) + ["[BOS]", "[EOS]", "[UNK]", "[PAD]"]
        self.indices = {w: i for i, w in enumerate(self.words)}
        self.nltk_tokenizer = TreebankWordTokenizer()

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item: str) -> int:
        if item in self.indices:
            return self.indices[item]
        else:
            return self.indices["[UNK]"]

    def __call__(self, batch) -> Dict[str, torch.LongTensor]:
        """
        Problem 2e: Implement this function.

        Converts a batch of examples, represented as raw text, into a
        tensor format compatible with the LSTMSentimentClassifier.

        :param batch: A batch of examples in raw form
        :return: The batch, in tensor form
        """
        # batch = {"text": [text1, text2], "label": [1, 0]}
        tensor_batch = {} 
        tokenized_texts = [
            self.postprocess(self.tokenize(Tokenizer.normalize(text)))
            for text in batch['text']
        ]
        text_lengths = [len(tokens) for tokens in tokenized_texts]
        max_len = max(text_lengths)

        tensor_batch['lengths'] = torch.tensor(text_lengths)
        tensor_batch['label'] = torch.tensor(batch['label'])
        tensor_batch['text'] = torch.tensor(
            [
                [self.indices[token] for token in tokens] + [self.indices['[PAD]']] * (max_len - len(tokens))
                for tokens in tokenized_texts
            ]
        )
        return tensor_batch

    @staticmethod
    def normalize(text: str) -> str:
        """
        Removes HTML tags from a text.

        :param text: A text, represented as a raw string
        :return: The text, with HTML tags removed and whitespace
            collapsed
        """
        soup = BeautifulSoup(text, "html.parser")
        return " ".join(soup.get_text(separator=" ").split())

    def tokenize(self, text: str) -> List[str]:
        """
        Problem 2d: Implement this function.

        Splits a text into tokens using self.nltk_tokenizer and NLTK's
        sent_tokenize function.

        :param text: A text, represented as a raw string
        :return: The text, split into a list of tokens
        """
        return sum([self.nltk_tokenizer.tokenize(sent) for sent in sent_tokenize(text)], [])

    def postprocess(self, tokens: List[str]) -> List[str]:
        """
        Problem 2d: Implement this function.

        Adds [BOS] and [EOS] to a list of tokens and replaces unknown
        tokens with [UNK].

        :param tokens: A list of tokens
        :return: The post-processed tokens
        """
        return [token if token in self.indices else '[UNK]' for token in ['[BOS]'] + tokens + ['[EOS]']] 
