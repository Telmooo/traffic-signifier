from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import numpy as np
from tqdm import tqdm

from models.common import Hyperparameters, ScoreHistory
from utils.utils import progress_bar


class BasicModel:
    def __init__(self, model, pretrained, n_classes, hyperparameters: Hyperparameters):
        self.model_name = model
        self.model = models.densenet201(pretrained=pretrained)

        self.model.classifier = nn.Linear(
            in_features=1920, out_features=n_classes, bias=True)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=hyperparameters['learning_rate'],
                                         momentum=hyperparameters['momentum']
                                         )

        self.metric_scorer = accuracy_score

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint)

    def freeze_feature_layer(self):
        self.model.features.requires_grad_(False)

    def unfreeze_feature_layer(self):
        self.model.features.requires_grad_(True)

    def epoch_iter(self, dataloader, is_train: bool = True):
        num_batches = len(dataloader)

        total_loss = 0.0
        preds = []
        true_labels = []

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(is_train):
            for _batch, (X, y) in enumerate(tqdm(dataloader)):
                labels = y["labels"]
                X, y = X.to(self.device), labels.to(self.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

                probs = F.softmax(pred, dim=1)
                final_pred = torch.argmax(probs, dim=1)

                preds.extend(final_pred.cpu().numpy())
                true_labels.extend(y.cpu().numpy())

        return total_loss / num_batches, self.metric_scorer(true_labels, preds)

    def train(self, num_epochs: int, train_dataloader, validation_dataloader, verbose=True, out_dir: str = "./"):
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

            train_loss, train_acc = self.epoch_iter(train_dataloader)
            val_loss, val_acc = self.epoch_iter(
                validation_dataloader, is_train=False)

            print(
                f"Training loss: {train_loss:.3f} \t Training accuracy: {train_acc:.3f}\n"
                f"Validation loss: {val_loss:.3f} \t Validation accuracy: {val_acc:.3f}\n"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_accuracy = val_acc
                save_dict = {"model": self.model.state_dict(
                ), "optimizer": self.optimizer.state_dict(), "epoch": epoch}
                torch.save(
                    save_dict, f"{out_dir}/{self.model_name}_best_model.pth")

            if verbose:
                save_dict = {"model": self.model.state_dict(
                ), "optimizer": self.optimizer.state_dict(), "epoch": epoch}
                torch.save(
                    save_dict, f"{out_dir}/{self.model_name}_latest_model.pth")

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
        test_loss, test_acc = self.epoch_iter(test_dataloader, is_train=False)
        return test_loss, test_acc
