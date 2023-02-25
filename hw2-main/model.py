"""
Code for Problem 3 of HW 2.
"""
import torch
import torch.nn as nn

from embeddings import Embeddings


class LSTMSentimentClassifier(nn.Module):
    """
    Problems 3b, 3c, and 3d: Complete the implementation of this class
    based on the docstrings and the usage examples in the problem set.

    This class defines the LSTM model architecture used for sentiment
    analysis in this assignment.
    """

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int):
        """
        Problem 3b: Implement this function.

        Constructs an LSTMSentimentClassifier by creating the modules
        that comprise it.

        :param vocab_size: The number of word embeddings to use in the
            model, including [BOS], [EOS], [UNK], and [PAD]
        :param embedding_size: The size of the word embeddings
        :param hidden_size: The size of the hidden state vector
        """
        super(LSTMSentimentClassifier, self).__init__()

        # Replace the "None"s below with the appropriately initialized
        # modules
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 2)

    def load_pretrained_embeddings(self, embeddings: Embeddings):
        """
        Problem 3c: Implement this function.

        Loads pre-trained embeddings into self.embeddings. The last 4
        rows of self.embeddings should be left unchanged, since they
        represent [BOS], [EOS], [UNK], and [PAD].

        :param embeddings: The pre-trained embeddings
        """
        embedding_weights = torch.cat([
            torch.tensor(embeddings.vectors).float(), 
            self.embeddings.weight[-4:, :]
        ])
        self.embeddings = nn.Embedding.from_pretrained(embedding_weights)

    def forward(self, text: torch.LongTensor, lengths: torch.LongTensor) -> \
            torch.FloatTensor:
        """
        Problem 3d: Implement this function.

        The model's forward method. The model should encode a sequence
        of hidden states for each text in an input batch, and predict a
        label for each text by decoding the last hidden state. Note that
        the last hidden state is not necessarily in the same position
        for each text in a batch, since the texts might have different
        lengths.

        :param text: A batch of input texts, represented as a matrix of
            indices. Shape: (batch size, maximum text length)
        :param lengths: The length of each text in the batch, including
            [BOS] and [EOS]. Shape: (batch size,)
        :return: The logit scores predicted for each text in the batch.
            Shape: (batch_size, 2)
        """
        hidden_states, _ = self.lstm(self.embeddings(text))
        self.hidden_states = torch.cat([
            hidden_states[i, j-1, :].reshape(1,-1) 
            for i,j in enumerate(lengths)
        ])
        outputs = self.linear(self.hidden_states)
        return outputs

        
