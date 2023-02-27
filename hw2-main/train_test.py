"""
Code for Problem 4 of HW 2.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
from model import LSTMSentimentClassifier


def evaluate(model: LSTMSentimentClassifier, test_data: Dataset,
             batch_size: int = 32) -> float:
    """
    Problem 4b: Complete the implementation of this function. You may
    write code anywhere in the function body. You must remove the raise
    statement on line 38.

    Evaluate an LSTM sentiment classifier on a testing dataset. In order
    to save memory, the evaluation should occur in mini-batches, such
    that the model is only being evaluated on a few examples at a time.

    :param model: The model being evaluated
    :param test_data: The data to evaluate the model with
    :param batch_size: The size of the mini-batches

    :return: The accuracy of the model on test_data, defined as the
        total proportion of examples (between 0 and 1) that have been
        classified correctly
    """
    model.eval()
    correct = 0 
    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size), position=0, leave=True):
            batch = test_data[i:i + batch_size]
            output = model(batch['text'], batch['lengths'])
            correct += torch.sum(output.argmax(axis=1) == batch['label'])
            
    return correct * 1.0 / len(test_data)


def train(model: LSTMSentimentClassifier, train_data: Dataset,
          val_data: Dataset, batch_size: int = 32, max_epochs: int = 5,
          patience: int = 2, lr: float = .01, filename: str = "model.pt", history_filename=None):
    """
    Problem 4c: Complete the implementation of this function. You may
    write code anywhere in the function body, though comments have been
    provided to guide the placement of your code. You must remove the
    raise statements on lines 74 and 82.

    :param model: The model to be trained
    :param train_data: The data to train the model on
    :param val_data: The data to use for early stopping
    :param batch_size: The size of the mini-batches for training and
        evaluation on validation data
    :param max_epochs: The maximum number of times ("epochs") to loop
        through all the training data
    :param patience: If this many consecutive epochs have passed without
        observing the best validation accuracy so far, then end training
        even if the maximum number of epochs has not yet been reached
    :param lr: The learning rate to use for Adam
    :param filename: The name of the file to save the model to
    """
    loss_function = nn.CrossEntropyLoss()
    adam = optim.Adam(model.parameters(), lr=lr)
    history = []
    best_val = -1
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    for epoch in range(max_epochs):
        print("Epoch {} of {}".format(epoch + 1, max_epochs))

        # Training code
        print("Training...")
        model.train()
        if torch.cuda.is_available():
            model = mode.to(device)
        for i in tqdm(range(0, len(train_data), batch_size),  position=0, leave=True):
            adam.zero_grad()
            batch = train_data[i:i + batch_size]
            if torch.cuda.is_available():
                batch['text'] = batch['text'].to(device)
                batch['label'] = batch['label'].to(device)
            # TODO: Write your training code here
            loss = loss_function(model(batch['text'], batch['lengths']), batch['label'])
            loss.backward()
            adam.step()

        # Test on validation data
        print("Evaluating on validation data...")
        val_acc = evaluate(model, val_data, batch_size=batch_size)
        print("Validation accuracy: {:.3f}".format(val_acc))

        # TODO: Write your early stopping code here
        history.append(val_acc)
        if len(history) >= patience and max(history[-patience:]) <= best_val:
            break 
        if history_filename:
            pd.DataFrame({'val acc': history}).to_csv(history_filename)
        if val_acc > best_val:
            torch.save(model.state_dict(), filename)
        best_val = max(best_val, val_acc)
        
    model.load_state_dict(torch.load(filename))
      
    
