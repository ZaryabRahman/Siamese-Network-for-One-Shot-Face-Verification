"""
custom Dataset and Data Loading for Siamese Network Training

this module defines the `SiameseTripletDataset` class, which is a custom PyTorch Dataset.
Its primary responsibility is to load the AT&T Faces dataset and generate triplets of images
(anchor, positive, negative) on-the-fly. This is essential for training a Siamese Network
with Triplet Loss.

  anchor: a base image from a specific person.
  positive: a different image of the same person as the anchor.
  negative: an image of a different person.

the dataset ensures that for each sample, the network receives these three images, which
are then used by the Triplet Loss function to learn a meaningful embedding space.

"""

import os
import random
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

class SiameseTripletDataset(Dataset):
    """
    a custom PyTorch Dataset for generating image triplets.

    this dataset wraps a torchvision.datasets.ImageFolder instance and, for each
    item retrieval, it returns an anchor image, a positive image (from the same class
    as the anchor), and a negative image (from a different class).

    attributes:
        image_folder_dataset (ImageFolder): The underlying dataset containing all images,
                                            organized in class-specific folders.
        transform (transforms.Compose): A sequence of transformations to be applied to each image.
        class_to_indices (Dict[int, List[int]]): A mapping from class labels (integers) to a list
                                                 of sample indices belonging to that class.
    """

    def __init__(self, image_folder_dataset: ImageFolder, transform: transforms.Compose = None) -> None:
        """
        initializes the SiameseTripletDataset.

        this constructor pre-processes the dataset to create a mapping from each class
        to the list of image indices associated with it. This mapping is crucial for
        efficiently sampling positive and negative images.

        args:
            image_folder_dataset (ImageFolder): An instance of ImageFolder, which should have
                                                already loaded the image dataset from a directory.
            transform (transforms.Compose, optional): A PyTorch transform to apply to each
                                                      image when it is loaded. Defaults to None.
        """
        super().__init__()
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform

        # pre-computation for efficient triplet sampling 
        self.class_to_indices: Dict[int, List[int]] = self._generate_class_to_indices_map()
        self.class_list = list(self.class_to_indices.keys())

    def _generate_class_to_indices_map(self) -> Dict[int, List[int]]:
        """
        creates a dictionary that maps each class label to a list of sample indices.

        this helper method iterates through the entire dataset once to build a mapping that
        allows for quick lookups of all images belonging to a specific class. This is
        a performance optimization to avoid searching the dataset every time a positive
        or negative sample is needed.

        returns:
            Dict[int, List[int]]: A dictionary where keys are class labels and values are
                                  lists of indices for samples of that class.
        """
        class_map = {}
        # self.image_folder_dataset.samples is a list of (image_path, class_index)
        for idx, (_, class_label) in enumerate(self.image_folder_dataset.samples):
            if class_label not in class_map:
                class_map[class_label] = []
            class_map[class_label].append(idx)
        return class_map

    def __len__(self) -> int:
        """
        returns the total number of samples (images) in the dataset.

        this is required by the PyTorch Dataset base class. The length is simply the total
        number of individual images in the wrapped ImageFolder dataset.

        returns:
            int: the total number of images in the dataset.
        """
        return len(self.image_folder_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        retrieves a single triplet (anchor, positive, negative) from the dataset.

        this is the core method of the Dataset class. Given an index, it performs the
        following steps:
          Retrieves the anchor image and its class label using the given index.
          Randomly samples a positive image: another image from the same class.
          Randomly samples a negative image: an image from a different class.
          Loads and transforms all three images.

        args:
            index (int): The index of the anchor image to retrieve.

        returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the transformed
                                                             anchor, positive, and negative images
                                                             as PyTorch tensors.
        """
        
        anchor_path, anchor_label = self.image_folder_dataset.samples[index]

        positive_indices = self.class_to_indices[anchor_label]
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(positive_indices)
        positive_path, _ = self.image_folder_dataset.samples[positive_index]

        negative_label = random.choice(self.class_list)
        while negative_label == anchor_label:
            negative_label = random.choice(self.class_list)
        negative_index = random.choice(self.class_to_indices[negative_label])
        negative_path, _ = self.image_folder_dataset.samples[negative_index]
        anchor_img = Image.open(anchor_path).convert("L")
        positive_img = Image.open(positive_path).convert("L")
        negative_img = Image.open(negative_path).convert("L")

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img
