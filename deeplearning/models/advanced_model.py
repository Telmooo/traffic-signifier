from sklearn.metrics import accuracy_score
from torchmetrics import Accuracy
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import numpy as np
from tqdm import tqdm

from models.common import Hyperparameters, ScoreHistory
from utils.utils import progress_bar


class AdvancedModel:
    def __init__(self, model : str, pretrained : bool, n_classes : int, hyperparameters: Hyperparameters):
        self.model_name = model
        self.model = models.densenet201(pretrained=pretrained)
        self.n_classes = n_classes
        self.model.classifier = nn.Linear(
            in_features=1920, out_features=self.n_classes, bias=True)

        self.loss_fn = nn.BCELoss()

        self.optimizer = torch.optim.Adam(
                                            self.model.parameters(),
                                            lr=hyperparameters['learning_rate'],
                                            betas=hyperparameters['betas'],
                                            weight_decay=hyperparameters['weight_decay'],
                                            amsgrad=hyperparameters['amsgrad']
                                         )

        self.metric_scorer = Accuracy(
            threshold=0.5,
            num_classes=self.n_classes,
            average="macro"
        )

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
            for _i, (X, y) in enumerate(tqdm(dataloader)):
                
                labels = y["labels"]
                
                X, y = X.to(self.device), labels.to(self.device);
                # print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY\n", y)
                # labels = y["labels"]

                # X, y = X.to(self.device), labels.to(self.device)
                # print(y)

                pred = self.model(X)

                # sigmoid activation to get values between 0 and 1
                sig = nn.Sigmoid()
                
                # print(y.dtype)
                loss = self.loss_fn(sig(pred), y.float())
                
                total_loss += loss.item()
                
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                preds.extend(pred.detach().cpu().numpy())
                true_labels.extend(y.detach().cpu().numpy())

        print(torch.tensor(np.array(true_labels)))
        return total_loss / num_batches, self.metric_scorer(torch.tensor(np.array(preds)), torch.tensor(np.array(true_labels)) )

    def train(self, num_epochs: int, train_dataloader, validation_dataloader, verbose=True, out_dir: str = "./"):
        train_history = ScoreHistory(loss=[], metric=[])
        val_history = ScoreHistory(loss=[], metric=[])

        best_val_loss = np.inf
        best_accuracy = 0

        print(
            f"Starting training...\n"
            f"Model: {self.model_name}\tDevice: {self.device}\n"
            f"Total Epochs: {num_epochs}\n"
        )

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}")

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

        progress_bar(epoch + 1, num_epochs, finished=True)
        print(
            f"Finished training...\n"
            f"Best loss: {best_val_loss}\tAccuracy on best loss: {best_accuracy}\n"
        )

        return train_history, val_history

    def test(self, test_dataloader):
        test_loss, test_acc = self.epoch_iter(test_dataloader, is_train=False)
        return test_loss, test_acc
