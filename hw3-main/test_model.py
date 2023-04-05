import pickle
import sys
import os
import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments

from train_model import preprocess_dataset, compute_metrics


def init_tester(directory: str) -> Trainer:
    """
    Prolem 1f: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """
    model = BertForSequenceClassification\
        .from_pretrained(directory, num_labels=2)
    return Trainer(model=model,compute_metrics=compute_metrics)

if __name__ == "__main__":
    model_name = "prajjwal1/bert-tiny"

    # checkpoint = 'checkpoints/checkpoint.no-bitfit.04-05-2023.05-07-43'
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # Set up tester
    tester = init_tester(checkpoint)

    # Test
    results = tester.predict(imdb["test"])
    with open(f"outputs/{os.path.basename(checkpoint)}.test_results.p", "wb") as f:
        pickle.dump(results, f)
