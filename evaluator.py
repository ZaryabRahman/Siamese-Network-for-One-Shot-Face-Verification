"""

Author: Zaryab Rahman
Data: 20/9/25

"""




"""
model evaluation module

this module defines the `ModelEvaluator` class, which is designed to assess the performance
of the trained Siamese Network. While the training process minimizes a loss function (Triplet Loss),
this loss value itself is not very intuitive for understanding real-world performance.

The evaluator computes more interpretable metrics:
  by finding an optimal distance threshold, it calculates the percentage of
    image pairs that are correctly classified as "same" or "different".
  a plot that visualizes the diagnostic
    ability of a binary classifier system as its discrimination threshold is varied.
  the area under the ROC curve. It provides an aggregate
    measure of performance across all possible classification thresholds. An AUC of 1.0
    represents a perfect model, while an AUC of 0.5 represents a model with no discriminative
    power (equivalent to random guessing).

to perform this evaluation, a different type of dataset is needed: one that provides pairs
of images along with a binary label (1 for same class, 0 for different class).
"""

from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import numpy as np

# A new dataset class is needed for evaluation. We define it here for convenience.
from dataset import SiameseTripletDataset # We can reuse the main dataset class and adapt
from torch.utils.data import Dataset
from PIL import Image
import random

class SiamesePairDataset(Dataset):
    """
    a Dataset class that generates pairs of images and a binary label.
    label 1  the images are from the same person.
    label 0 the images are from different people.
    this is specifically for evaluation purposes.
    """
    def __init__(self, image_folder_dataset, transform=None):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        self.class_to_indices = self._generate_class_to_indices_map()
        self.samples = self.image_folder_dataset.samples

    def _generate_class_to_indices_map(self):
        class_map = {}
        for idx, (_, class_label) in enumerate(self.image_folder_dataset.samples):
            if class_label not in class_map:
                class_map[class_label] = []
            class_map[class_label].append(idx)
        return class_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        anchor_path, anchor_label = self.samples[index]
        
        # randomly decide to create a positive (same) or negative (different) pair
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            positive_indices = self.class_to_indices[anchor_label]
            other_index = index
            while other_index == index:
                other_index = random.choice(positive_indices)
            label = 1.0
        else:
            negative_label = random.choice(list(self.class_to_indices.keys()))
            while negative_label == anchor_label:
                negative_label = random.choice(list(self.class_to_indices.keys()))
            other_index = random.choice(self.class_to_indices[negative_label])
            label = 0.0
            
        other_path, _ = self.samples[other_index]

        anchor_img = Image.open(anchor_path).convert("L")
        other_img = Image.open(other_path).convert("L")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            other_img = self.transform(other_img)

        return anchor_img, other_img, torch.tensor(label, dtype=torch.float32)

class ModelEvaluator:
    """
    
    Handles the evaluation of the 
    Siamese Network using 
    accuracy, 
    ROC, and 
    AUC metrics.
    
    """
    def __init__(self, model: nn.Module, test_loader: DataLoader, device: str) -> None:
        """
        initializes the ModelEvaluator.

        args:
            model (nn.Module): The trained Siamese Network model to be evaluated.
            test_loader (DataLoader): DataLoader for the test dataset, which should yield
                                      pairs of images and a binary label.
            device (str): The device to run evaluation on ('cuda' or 'cpu').
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.to(self.device)
        self.model.eval() # ensure the model is in evaluation mode

    def _compute_distances_and_labels(self) -> Tuple[List[float], List[int]]:
        """
        computes the Euclidean distances for all pairs in the test set.

        this private method iterates through the entire test dataloader, performs a
        forward pass for each pair to get their embeddings, computes the distance
        between them, and collects all distances and their corresponding ground truth labels.

        returns:
            Tuple[List[float], List[int]]: A tuple containing two lists:
                - A list of all computed Euclidean distances.
                - A list of the corresponding ground truth labels (0 or 1).
        """
        distances = []
        ground_truth_labels = []
        with torch.no_grad():
            for img1, img2, label in self.test_loader:
                img1, img2 = img1.to(self.device), img2.to(self.device)
                
                # we need to adapt the forward pass for evaluation (pairs instead of triplets)
                # let's create a temporary method to handle this cleanly.
                emb1 = self.model.forward_once(img1)
                emb2 = self.model.forward_once(img2)
                
                dist = F.pairwise_distance(emb1, emb2).cpu().numpy()
                distances.extend(dist)
                ground_truth_labels.extend(label.cpu().numpy())
        return distances, ground_truth_labels

    def evaluate(self) -> Dict:
        """
        performs the full evaluation and returns a dictionary of metrics.

        This is the main public method of the class. It orchestrates the entire
        evaluation process:
          Computes all distances and labels.
          Calculates the ROC curve and AUC.
          Finds the best threshold and the corresponding accuracy.

        returns:
            Dict: A dictionary containing the evaluation results:
                  'accuracy', 'best_threshold', 'roc_auc', 'fpr', 'tpr'.
        """
        print("\nStarting model evaluation...")
        distances, labels = self._compute_distances_and_labels()
        
        labels_int = np.array(labels).astype(int)
        
        # invert distances for roc_curve: it expects scores where higher is better.
        # for distances, a lower value means "more likely to be the same class".
        # so, we can use (1 - normalized_distance) or simply (-distance).
        scores = -np.array(distances)
        
        fpr, tpr, roc_thresholds = roc_curve(labels_int, scores)
        roc_auc = auc(fpr, tpr)
        
        # find the best threshold for accuracy
        # we check every possible threshold on the original distances
        accuracy_thresholds = np.linspace(min(distances), max(distances), 100)
        accuracies = []
        for thresh in accuracy_thresholds:
            preds = (np.array(distances) < thresh).astype(int)
            acc = np.mean(preds == labels_int)
            accuracies.append(acc)
            
        best_accuracy = max(accuracies)
        best_threshold = accuracy_thresholds[np.argmax(accuracies)]

        print("Evaluation Results:")
        print(f"  Best Accuracy: {best_accuracy:.4f}")
        print(f"  At Distance Threshold: {best_threshold:.4f}")
        print(f"  Area Under ROC Curve (AUC): {roc_auc:.4f}")
        
        return {
            "accuracy": best_accuracy,
            "best_threshold": best_threshold,
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr
        }
