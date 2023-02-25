from train_test import train
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from nltk.tokenize import TreebankWordTokenizer
from datetime import datetime

from embeddings import Embeddings
from model import LSTMSentimentClassifier
from tokenizer import Tokenizer
from train_test import evaluate, train


def run(lr=0.01, batch_size=32, testing=True, use_pretrained=True):
    
    # get tokenizer and model 
    glove = Embeddings.from_file("data/glove_300d.txt")
    tokenizer = Tokenizer(glove.words)
    model = LSTMSentimentClassifier(len(tokenizer), 300, 10)
    if use_pretrained:
        model.load_pretrained_embeddings(glove)
    
    # load dataset
    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    if "unsupervised" in imdb:
        del imdb["unsupervised"]
    train_data = imdb["train"].with_transform(tokenizer)
    val_data = imdb["val"].with_transform(tokenizer)
    test_data = imdb["test"].with_transform(tokenizer)
    if testing:
        train_data_small = train_data.shard(num_shards=100, index=0)
        val_data_small = train_data_small
        test_data_small = train_data_small
        
    # time = datetime.now().strftime("%m-%d-%Y_%H:%M")
    embeddings_used = 'random' if not use_pretrained else 'glove' 
    
    train(model, train_data, val_data, 
          batch_size=batch_size, 
          max_epoch=50,
          patience=5, 
          lr=lr, 
          filename= f"model_{embeddings_used}_{lr}_{batch_size}.pt"
   )

if __init__ == "__main__":
      run(lr=0.01, batch_size=32)

