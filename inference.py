#
"""
inf script for face verification 

this script provides a simple command-line
interface to use the trained Siamese Network
for verifying whether two face images 
belong to the same person. It loads the best-trained
model weights and the optimal distance 
threshold found during evaluation, processes the
two input images, and outputs a prediction.

this serves as a practical demonstration of the model's application.

how to run:
    python inference.py --img1 "path/to/first/image.jpg" --img2 "path/to/second/image.jpg"
"""

import argparse
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

import config
from model import SiameseResNet

def preprocess_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    """
    loads an image from a path, converts it to grayscale, and applies transformations.

    args:
        image_path (str): The file path to the image.
        transform (transforms.Compose): The sequence of transformations to apply.

    returns:
        torch.Tensor: The processed image tensor, ready to be fed into the model.
    """
    try:
        img = Image.open(image_path).convert("L")  # convert to grayscale
    except FileNotFoundError:
        print(f"Error: Image not found at path: {image_path}")
        exit(1)
        
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)

def run_inference(model_path: str, img1_path: str, img2_path: str, threshold: float):
    """
    performs face verification on a pair of 
    images using the trained Siamese Network.

    args:
        model_path (str): Path to the saved model state dictionary (.pth file).
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        threshold (float): The distance threshold for classification. Pairs with a distance
                           below this threshold are considered "same", and "different" otherwise.
    """
    print("\nFace Verification Inference")
    
    
    device = config.DEVICE
    print(f"Using device: {device.upper()}")

    inference_transform = transforms.Compose([
        transforms.Resize(config.DATA_TRANSFORMS['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.DATA_TRANSFORMS['mean'], std=config.DATA_TRANSFORMS['std'])
    ])

    print(f"Loading model from: {model_path}")
    model = SiameseResNet(
        embedding_dim=config.MODEL_CONFIG['embedding_dim'],
        use_pretrained=False # pretrained weights are already part of the state_dict
    ).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        print("Please run main.py to train and save the model first.")
        exit(1)
        
    model.eval() # set the model to evaluation mode

  
    print(f"Processing images: \n  - Image 1: {img1_path}\n  - Image 2: {img2_path}")
    img1_tensor = preprocess_image(img1_path, inference_transform).to(device)
    img2_tensor = preprocess_image(img2_path, inference_transform).to(device)
  
    with torch.no_grad():
        embedding1 = model.forward_once(img1_tensor)
        embedding2 = model.forward_once(img2_tensor)

    # Cclculate Euclidean distance
    distance = torch.nn.functional.pairwise_distance(embedding1, embedding2).item()

    
    print("\nResults")
    print(f"Computed Distance: {distance:.4f}")
    print(f"Decision Threshold: {threshold:.4f}")

    if distance < threshold:
        print("Prediction:  SAME PERSON")
    else:
        print("Prediction:  DIFFERENT PEOPLE")
    print("-" * 15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run face verification using a trained Siamese Network.")
    parser.add_argument("--img1", type=str, required=True, help="Path to the first image.")
    parser.add_argument("--img2", type=str, required=True, help="Path to the second image.")
    parser.add_argument("--model_path", type=str, default=config.MODEL_SAVE_PATH, help="Path to the trained model file.")
    parser.add_argument("--threshold", type=float, default=1.0, help="Distance threshold for classification.")
    
    args = parser.parse_args()
    
    run_inference(args.model_path, args.img1, args.img2, args.threshold)
