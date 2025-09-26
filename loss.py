"""
custom Loss Function for Metric Learning: Triplet Loss

this module defines the `TripletLoss` class, a custom loss function specifically designed
for training Siamese Networks and other models for metric learning. The primary goal of
Triplet Loss is to learn an embedding space where the distance between samples of the
same class is minimized, and the distance between samples of different classes is maximized.

the loss function operates on a triplet of embeddings:
- `anchor`: The embedding of a reference sample.
- `positive`: The embedding of another sample from the same class as the anchor.
- `negative`: The embedding of a sample from a different class than the anchor.

the loss is calculated as:
L(a, p, n) = max( d(a, p)^2 - d(a, n)^2 + margin, 0 )

where `d(x, y)` is the Euclidean distance between embeddings x and y, and `margin` is a
hyperparameter that enforces a minimum separation between positive and negative pairs.
By minimizing this loss, the model is trained to ensure that the anchor is closer to the
positive than it is to the negative, by at least the specified margin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    implements the Triplet Loss for learning discriminative embeddings.

    this loss function encourages the model to produce embeddings such that the distance
    between an anchor and a positive sample is smaller than the distance between the
    anchor and a negative sample, plus a margin.

    attributes:
        margin (float): The margin value. This is a hyperparameter that defines the
                        minimum desired separation between the positive and negative
                        pair distances.
    """

    def __init__(self, margin: float = 1.0) -> None:
        """
        initializes the TripletLoss module.

        args:
            margin (float, optional): The margin to be used in the loss calculation.
                                      Defaults to 1.0, a commonly used value.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        computes the Triplet Loss for a batch of triplets.

        args:
            anchor (torch.Tensor): The embedding vectors for the anchor samples.
                                   Shape: (N, embedding_dim).
            positive (torch.Tensor): The embedding vectors for the positive samples.
                                     Shape: (N, embedding_dim).
            negative (torch.Tensor): The embedding vectors for the negative samples.
                                     Shape: (N, embedding_dim).

        returns:
            torch.Tensor: A scalar tensor representing the mean Triplet Loss for the batch.
        """
        # F.pairwise_distance computes the distance between each pair of vectors
        # in the two input tensors.
        distance_positive = F.pairwise_distance(anchor, positive, p=2)

        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        # The core formula: loss = max(distance_positive - distance_negative + margin, 0)
        # We use F.relu which is equivalent to max(x, 0).
        losses = F.relu(distance_positive - distance_negative + self.margin)

        # we take the mean of the losses over the entire batch to get a single scalar
        # value that can be used for backpropagation.
        return torch.mean(losses)
