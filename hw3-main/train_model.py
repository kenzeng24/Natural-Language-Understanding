"""
Code for Problem 1 of HW 3.
"""
import pickle
import sys 
import torch 
from typing import Any, Dict
from datetime import datetime

import evaluate
import numpy as np
import optuna
import torch.nn as nn 
from datasets import Dataset, load_dataset, load_metric
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments, EvalPrediction
from transformers import EarlyStoppingCallback


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        data = self.data[index]
        encoding = self.tokenizer(
            data['text'], 
            truncation=True, 
            max_length=132,
            padding='max_length'
        )
        return dict(**encoding, **data)

    def __len__(self):
        return len(self.data)

    

def preprocess_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) \
        -> Dataset:
    """
    Problem 1d: Implement this function.

    Preprocesses a dataset using a Hugging Face Tokenizer and prepares
    it for use in a Hugging Face Trainer.

    :param dataset: A dataset
    :param tokenizer: A tokenizer
    :return: The dataset, prepreprocessed using the tokenizer
    """
    return CustomDataset(dataset, tokenizer)
    


def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> \
        BertForSequenceClassification:
    """
    Problem 1e: Implement this function.

    This function should be passed to your Trainer's model_init keyword
    argument. It will be used by the Trainer to initialize a new model
    for each hyperparameter tuning trial. Your implementation of this
    function should support training with BitFit by freezing all non-
    bias parameters of the initialized model.

    :param trial: This parameter is required by the Trainer, but it will
        not be used for this problem. Please ignore it
    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be loaded
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A newly initialized pre-trained Transformer classifier
    """
    model = BertForSequenceClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=2
    )
    if use_bitfit:
        for key, value in dict(model.named_parameters()).items():
            if 'bias' not in key:
                value.requires_grad = False 
    return model 


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_preds):
    metric = load_metric('accuracy')
    preds, labels = eval_preds
    preds = np.argmax(preds, axis=1)
    return metric.compute(predictions=preds, references=labels)


def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset,
                 use_bitfit: bool = False) -> Trainer:
    """
    Prolem 1f: Implement this function.

    Creates a Trainer object that will be used to fine-tune a BERT-tiny
    model on the IMDb dataset. The Trainer should fulfill the criteria
    listed in the problem set.

    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be fine-tuned
    :param train_data: The training data used to fine-tune the model
    :param val_data: The validation data used for hyperparameter tuning
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A Trainer used for training
    """

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=32,
        learning_rate=1e-3,
        disable_tqdm=False,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
    )
    # Create trainer object
    trainer = CustomTrainer(
        model=None,
        model_init=lambda: init_model(None, model_name, use_bitfit),
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    return trainer


def hyperparameter_search_settings() -> Dict[str, Any]:
    """
    Problem 1g: Implement this function.

    Returns keyword arguments passed to Trainer.hyperparameter_search.
    Your hyperparameter search must satisfy the criteria listed in the
    problem set.

    :return: Keyword arguments for Trainer.hyperparameter_search
    """
    # for non-bitfit: [3e-4, 1e-4, 5e-5, 3e-5, 2e-5]
    # for bitfit:     [3e-3, 1e-3, 3e-4, 1e-4, 5e-5]
    
    search_space = {
        "seed":[824],
        "num_train_epochs":[4],
        "learning_rate":[3e-4, 1e-4, 5e-5, 3e-5], 
        "per_device_train_batch_size":[8, 16, 32, 64, 128],
    }
    
    return dict(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=25,
        sampler= optuna.samplers.GridSampler(search_space)
        compute_objective=lambda metrics: metrics['eval_accuracy'],
    )


if __name__ == "__main__":  # Use this script to train your model
    
    use_bitfit = True  
    if len(sys.argv) > 1:
        use_bitfit = sys.argv[1] == 'True'
    
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset and create validation split
    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    del imdb["unsupervised"]
    # del imdb["test"]

    # Preprocess the dataset for the trainer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    imdb["train"] = preprocess_dataset(imdb["train"], tokenizer)
    imdb["val"] = preprocess_dataset(imdb["val"], tokenizer)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # Set up trainer
    trainer = init_trainer(model_name, imdb["train"], imdb["val"],
                           use_bitfit=use_bitfit)

    # Train and save the best hyperparameters
    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    bitfit_param = 'bitfit' if use_bitfit else 'no-bitfit'
    time = datetime.now().strftime("%m-%d-%Y.%H-%M-%S")
    with open(f"outputs/train_results.{bitfit_param}.{time}.pickle", "wb") as f:
        pickle.dump(best, f)
    trainer.save_model(f'checkpoints/checkpoint.{bitfit_param}.{time}')
    print(trainer.evaluate(imdb['test']))

