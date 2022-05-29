from typing import List, TypedDict
import sys

from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torchvision import models
import numpy as np
from tqdm import tqdm

from deeplearning.utils.utils import epoch_iter, progress_bar

class Hyperparameters(TypedDict, total=False):
    learning_rate: float
    momentum: float

class ScoreHistory(TypedDict):
    loss: List[float]
    metric: List[float]

class BasicModel(nn.Module):
    def __init__(self, model, pretrained, n_classes, hyperparameters: Hyperparameters):
        super(BasicModel, self).__init__()
        self.model_name = model
        self.model = models.densenet201(pretrained=pretrained)

        self.model.classifier = nn.Linear(in_features=1920, out_features=n_classes, bias=True)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        self.optimizer = torch.optim.SGD(self.model.parameters(),
            lr=hyperparameters['learning_rate'],
            momentum=hyperparameters['momentum']
        )

        self.metric_scorer = accuracy_score

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, mode: bool = True):
        self.model.train(mode)
        return super().train(mode)

    def eval(self):
        self.model.train(False)
        return super().eval()

    def forward(self, x):
        logits = self.model(x)
        return logits

    def freeze_feature_layer(self):
        self.model.features.requires_grad_(False)
    def unfreeze_feature_layer(self):
        self.model.features.requires_grad_(True)

    def epoch_iter(self, dataloader):
        num_batches = len(dataloader)

        total_loss = 0.0
        preds = []
        labels = []

        with torch.set_grad_enabled(self.training):
            for batch, (X, y) in enumerate(tqdm(dataloader)):
                labels = y["labels"]
                X, y = X.to(self.device), labels.to(self.device)


        
        

    def train(self, num_epochs: int, train_dataloader, validation_dataloader, verbose=True):
        train_history = ScoreHistory()
        val_history = ScoreHistory()

        best_val_loss = np.inf
        best_accuracy = 0

        print(
            f"Starting training...\n"
            f"Model: {self.model_name}\tDevice: {self.device}\n"
            f"Total Epochs: {num_epochs}\n"
        )

        for epoch in range(num_epochs):
            progress_bar(epoch, num_epochs)
            
            train_loss, train_acc = epoch_iter(train_dataloader)
            val_loss, val_acc = epoch_iter(validation_dataloader, is_train=False)

            sys.stdout.write(
                f"Training loss: {train_loss:.3f} \t Training accuracy: {train_acc:.3f}\n"
                f"Validation loss: {val_loss:.3f} \t Validation accuracy: {val_acc:.3f}\r"
            )
            sys.stdout.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_accuracy = val_acc
                save_dict = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "epoch": epoch}
                torch.save(save_dict, self.model_name + "_best_model.pth")

            if verbose:
                save_dict = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "epoch": epoch}
                torch.save(save_dict, self.model_name + "_latest_model.pth")

            train_history["loss"].append(train_loss)
            train_history["metric"].append(train_acc)
            
            val_history["loss"].append(val_loss)
            val_history["metric"].append(val_acc)

        progress_bar(epoch, num_epochs, finished=True)
        print(
            f"Finished training...\n"
            f"Best loss: {best_val_loss}\tAccuracy on best loss: {best_accuracy}\n"
        )

        return train_history, val_history

    def test(self, test_dataloader):
        test_loss, test_acc = epoch_iter(test_dataloader, self.model, self.loss_fn, self.metric_scorer, self.device, is_train=False)
        print(f"Testing loss: {test_loss}\nTesting accuracy: {test_acc}\n")
