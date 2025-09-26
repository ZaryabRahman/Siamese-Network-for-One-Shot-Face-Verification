"""

Author: Zaryab Rahman
Data: 24/9/25

"""




"""
main script 

it performs followling seq operations

data preparation
model initialization
training
evaluation
visualization

"""


import os
import random
from typing import List, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


import config
from dataset import SiameseTripletDataset
from model import SiameseResNet
from loss import TripletLoss
from trainer import ModelTrainer
from evaluator import ModelEvaluator, SiamesePairDataset
from visualize import plot_loss_history, plot_roc_curve


def create_subject_-based_split(dataset: ImageFolder, train_split_ratio: float = 0.8) -> Tuple[List[int], List[int]]:
    """
    splits the dataset indices based on subjects (classes) rather than individual images.

    this ensures that all images of a particular person 
    belong exclusively to either the
    training set or the test set. 
    this is a more robust evaluation method for face
    recognition and one-shot learning tasks.

    args:
        dataset (ImageFolder): The full ImageFolder dataset.
        train_split_ratio (float): The proportion of subjects to allocate to the training set.

    returns:
        Tuple[List[int], List[int]]: A tuple containing two lists of indices:
                                     - The first list for the training subset.
                                     - The second list for the testing subset.
    """


    class_indices = list(range(len(dataset.classes)))
    random.shuffle(class_indices)

    num_train_classes = int(len(class_indices) * train_split_ratio)
    train_class_labels = class_indices[:num_train_classes]
    test_class_labels = class_indices[num_train_classes:]

    train_indices = []
    test_indices = []

    for i, (_, label) in enumerate(dataset.samples):
        if label in train_class_labels:
            train_indices.append(i)
        else:
            test_indices.append(i)

    return train_indices, test_indices


def main_pipeline():
    """
    
    The main function 
    that runs the entire 
    project pipeline.
    
    """
    print("Advanced Siamese Network for One-Shot Learning")
    print(f"Using device: {config.DEVICE.upper()}")
    print("-" * 55)

    
    print("Preparing Data...")
    train_transform = transforms.Compose([
        transforms.Resize(config.DATA_TRANSFORMS['image_size']),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.DATA_TRANSFORMS['mean'], std=config.DATA_TRANSFORMS['std'])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(config.DATA_TRANSFORMS['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.DATA_TRANSFORMS['mean'], std=config.DATA_TRANSFORMS['std'])
    ])

    full_dataset = ImageFolder(root=config.DATA_DIR)

    #
    train_indices, test_indices = create_subject_based_split(full_dataset, train_split_ratio=0.8)
    
    train_subset = Subset(full_dataset, train_indices)
    test_subset = Subset(full_dataset, test_indices)

    #  custom datasets need access to the original dataset's structure
    # to create pairs/triplets correctly. So we create new ImageFolder-like objects.
    # this is a bit of a workaround for using ImageFolder with subsets.
    train_subset.dataset.transform = train_transform
    test_subset.dataset.transform = test_transform
    
    # for training, we need a dataset that can provide triplets
    train_triplet_dataset = SiameseTripletDataset(image_folder_dataset=train_subset.dataset, transform=train_transform)
    # remap indices for the Triplet dataset to work with the subset
    train_triplet_dataset.image_folder_dataset.samples = [full_dataset.samples[i] for i in train_indices]
    train_triplet_dataset.class_to_indices = train_triplet_dataset._generate_class_to_indices_map()


    test_pair_dataset = SiamesePairDataset(image_folder_dataset=test_subset.dataset, transform=test_transform)
    test_pair_dataset.image_folder_dataset.samples = [full_dataset.samples[i] for i in test_indices]
    test_pair_dataset.class_to_indices = test_pair_dataset._generate_class_to_indices_map()

    train_loader = DataLoader(train_triplet_dataset, batch_size=config.TRAINING_CONFIG['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_pair_dataset, batch_size=config.EVAL_CONFIG['batch_size'], shuffle=False, num_workers=4)
    print("Data preparation complete.")
    print("-" * 55)

    print("initializing Model, Loss, and Optimizer...")
    model = SiameseResNet(
        embedding_dim=config.MODEL_CONFIG['embedding_dim'],
        use_pretrained=config.MODEL_CONFIG['use_pretrained']
    ).to(config.DEVICE)

    loss_fn = TripletLoss(margin=config.TRAINING_CONFIG['triplet_margin'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.TRAINING_CONFIG['learning_rate'],
        weight_decay=config.TRAINING_CONFIG['weight_decay']
    )
    print("initialization complete.")
    print("-" * 55)

    
    writer = None
    if config.LOGGING_CONFIG['enable_logging']:
        writer = SummaryWriter(log_dir=config.LOGGING_CONFIG['log_dir'])
        print(f"TensorBoard logging enabled. Logs will be saved to: {config.LOGGING_CONFIG['log_dir']}")
        print("-" * 55)

    
    print("STEP 3: Starting Training Phase...")
    trainer_config = {
        'num_epochs': config.TRAINING_CONFIG['num_epochs'],
        'model_save_path': config.MODEL_SAVE_PATH
    }
    # for the trainer, we need a validation set. We can reuse the test set for simplicity,
    # or create a dedicated validation split from the training subjects. Here we reuse.
    trainer = ModelTrainer(model, loss_fn, optimizer, train_loader, train_loader, config.DEVICE, trainer_config, writer)
    loss_history = trainer.train()
    print("Training phase complete.")
    print("-" * 55)
    
  
    print("Starting Evaluation Phase...")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    evaluator = ModelEvaluator(model, test_loader, config.DEVICE)
    eval_results = evaluator.evaluate()
    print("Evaluation phase complete.")
    print("-" * 55)

    print("Generating Visualizations...")
    plot_loss_history(loss_history)
    plot_roc_curve(eval_results)
    print("Visualizations generated.")
    print("-" * 55)

    print("Project pipeline finished successfully!")


if __name__ == "__main__":
    main_pipeline()
