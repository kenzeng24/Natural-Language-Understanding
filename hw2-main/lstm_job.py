import sys
from train_test import train
from datasets import load_dataset
from embeddings import Embeddings
from model import LSTMSentimentClassifier
from tokenizer import Tokenizer
from train_test import evaluate, train


def train_lstm_model(lr=0.01, batch_size=32, testing=True, use_pretrained=True):
    
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
        train_data = train_data.shard(num_shards=100, index=0)
        val_data = train_data
        test_data = train_data
        
    # time = datetime.now().strftime("%m-%d-%Y_%H:%M")
    embeddings_used = 'random' if not use_pretrained else 'glove' 
    
    train(model, train_data, val_data, 
          batch_size=batch_size, 
          max_epochs=1,
          patience=1, 
          lr=lr, 
          filename= f"checkpoints/{embeddings_used}_{lr}_{batch_size}.pt", 
          history_filename = f"checkpoints/{embeddings_used}_{lr}_{batch_size}_history.csv", 
    )
    evaluate(model, test_data)
    
    
def main():
    pretrained = bool(sys.argv[1])
    lr = float(sys.argv[2])
    batch_size = int(sys.argv[3])
    train_lstm_model(use_pretrained=pretrained, lr=lr, batch_size=batch_size)
    

if __name__ == "__main__":
    
    main()

