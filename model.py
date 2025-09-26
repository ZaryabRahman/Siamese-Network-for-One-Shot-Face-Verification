from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models

class SiameseResNet(nn.Module):
    """
    a Siamese Network that uses a pre-trained ResNet-18 as its backbone.

    this network takes two images as input and computes an embedding vector for each.
    The distance between these two vectors in the embedding space can then be used to
    determine the similarity of the input images.

    attributes:
        resnet (nn.Module): The modified ResNet-18 backbone used for feature extraction.
    """

    def __init__(self, embedding_dim: int = 128, use_pretrained: bool = True) -> None:
        """
        initializes the SiameseResNet model.

        args:
            embedding_dim (int, optional): The desired dimensionality of the output
                                           embedding vector. Defaults to 128.
            use_pretrained (bool, optional): Whether to use weights pre-trained on ImageNet
                                             for the ResNet-18 backbone. Defaults to True.
        """
        super(SiameseResNet, self).__init__()

        if use_pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        self.resnet = models.resnet18(weights=weights)

        
        # the original ResNet `conv1` is designed for 3-channel RGB images. We must adapt it
        # for our 1-channel grayscale images. We do this by creating a new Conv2d layer
        # and initializing its weights by averaging the weights of the original layer
        # across the input channel dimension. This is a common practice to transfer
        # color-based learning to grayscale.
        original_conv1_weights = self.resnet.conv1.weight.data
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )
        self.resnet.conv1.weight.data = torch.mean(original_conv1_weights, dim=1, keepdim=True)

        
        # We replace the original final fully-connected layer (which was for 1000-class
        # ImageNet classification) with our own sequence of layers. This new "head"
        # will project the features extracted by the ResNet body into the target
        # embedding space of `embedding_dim`.
        num_features_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features=num_features_in, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=embedding_dim),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        performs a single forward pass to generate an embedding for one image.

        this method processes a single batch of images through the ResNet backbone and
        the custom embedding head to produce their corresponding embedding vectors.

        args:
            x (torch.Tensor): A batch of input images with shape (N, 1, H, W), where N is the
                              batch size, 1 is the number of channels (grayscale), and H, W
                              are the height and width.

        returns:
            torch.Tensor: The resulting embedding vectors of shape (N, embedding_dim).
        """
        output = self.resnet(x)
        return output

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        performs the full forward pass for a triplet of images.

        this is the main forward method called during training. It processes the anchor,
        positive, and negative images through the same underlying network (`forward_once`)
        to get their respective embeddings.

        args:
            anchor (torch.Tensor): A batch of anchor images.
            positive (torch.Tensor): A batch of positive images.
            negative (torch.Tensor): A batch of negative images.

        returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the
                embedding vectors for the anchor, positive, and negative images.
        """
        # the core of the Siamese architecture: processing all inputs
        # through the identical, weight-shared network.
        embedding_anchor = self.forward_once(anchor)
        embedding_positive = self.forward_once(positive)
        embedding_negative = self.forward_once(negative)

        return embedding_anchor, embedding_positive, embedding_negative
