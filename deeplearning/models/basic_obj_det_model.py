import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from tqdm import tqdm

from models.common import Hyperparameters, ScoreHistory
from utils.utils import progress_bar

class BasicObjDectModel:
    def __init__(self, model: str, n_classes: int, hyperparameters: Hyperparameters) -> None:
        self.model_name = model
        self.model = models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )

        self.n_classes = n_classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels=in_features, num_classes=self.n_classes)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=hyperparameters['learning_rate'],
                                         momentum=hyperparameters['momentum']
                                         )

    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint)
    
    def freeze_backbone_layer(self):
        self.model.backbone.requires_grad_(False)

    def unfreeze_backbone_layer(self):
        self.model.backbone.requires_grad_(True)

    def epoch_iter(self, dataloader, is_train : bool = True):
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
                X = list(x.to(self.device) for x in X)
                y = [ { k: v.to(self.device) for k, v in annotation.items() } for annotation in y]

                loss_dict = self.model(X, y)

                print("LOSS_DICT")
                print(loss_dict)

                exit(0)
                losses = sum(loss for loss in loss_dict.values())
                
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

    def train(self, num_epochs : int, train_dataloader, validation_dataloader, verbose=True, out_dir : str = "./"):
        train_history = ScoreHistory(loss=[], metric=[])
        val_history = ScoreHistory(loss=[], metric=[])

        best_val_loss = np.inf
        best_metric = 0

        print(
            f"Starting training...\n"
            f"Model: {self.model_name}\tDevice: {self.device}\n"
            f"Total Epochs: {num_epochs}\n"
        )

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}")

            progress_bar(epoch, num_epochs)

            train_loss, train_metric = self.epoch_iter(train_dataloader)
        