if __name__ == "__main__":
    import os
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    import numpy as np

    from models.basic_obj_det_model import BasicObjDectModel
    from models.common import Hyperparameters
    from utils.dataloaders import RoadSignDataset
    
    DATA_DIR = "../data/"
    IMG_DIR = DATA_DIR + "/images/"
    ANNOTATION_DIR = DATA_DIR + "/annotations/"
    SPLITS_DIR = DATA_DIR + "/dl-split/"
    OUT_DIR = "./out"

    os.makedirs(OUT_DIR, exist_ok=True)

    train_split = []
    test_split = []

    with open(SPLITS_DIR + "/train.txt") as train_f:
        train_split = [line.strip("\n") for line in train_f.readlines()]

    with open(SPLITS_DIR + "/test.txt") as test_f:
        test_split = [line.strip("\n") for line in test_f.readlines()]

    # Create datasets
    training_data = RoadSignDataset(
        images_filenames=train_split,
        images_directory=IMG_DIR,
        annotations_directory=ANNOTATION_DIR,
        is_train=True,
        obj_detect=True,
        multilabel=True
    )

    testing_data = RoadSignDataset(
        images_filenames=train_split,
        images_directory=IMG_DIR,
        annotations_directory=ANNOTATION_DIR,
        is_train=False,
        obj_detect=True,
        multilabel=True
    )

    # Create train and validation from train split
    np.random.seed(42)

    train_indices = list(range(len(training_data)))
    np.random.shuffle(train_indices)
    TRAIN_VAL_SPLIT = int(np.floor(0.2 * len(train_indices)))

    train_idx, val_idx = train_indices[TRAIN_VAL_SPLIT:], train_indices[:TRAIN_VAL_SPLIT]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # DataLoaders
    BATCH_SIZE = 4
    NUM_WORKERS = 2

    train_dataloader = DataLoader(
        dataset=training_data,
        sampler=train_sampler,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        drop_last=True,
        collate_fn=training_data.collate_fn
    )

    val_dataloader = DataLoader(
        dataset=training_data,
        sampler=val_sampler,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        drop_last=True,
        collate_fn=training_data.collate_fn
    )

    test_dataloader = DataLoader(
        dataset=testing_data,
        batch_size=1,
        num_workers=NUM_WORKERS,
        drop_last=False,
        shuffle=False,
        collate_fn=testing_data.collate_fn
    )

    # Model
    hyperparameters = Hyperparameters(
        learning_rate=1e-3,
        momentum=0
    )

    model = BasicObjDectModel(
        model="basic_obj_dect",
        n_classes=4,
        hyperparameters=hyperparameters
    )

    # Training phase
    NUM_EPOCHS = 10
    
    model.freeze_backbone_layer()

    model.train(
        num_epochs=NUM_EPOCHS,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        verbose=True,
        out_dir=OUT_DIR
    )