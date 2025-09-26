"""
this file contains all the hyperparameters, paths, and settings used throughout the project.
By centralizing the configuration, we can easily modify and experiment with different settings
without having to change the core logic of the application. This promotes modularity and
maintainability.
"""

import torch
from typing import Dict, Any

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR: str = "./data/att_faces"  # path to the root directory of the AT&T Faces dataset.
MODEL_SAVE_PATH: str = "./siamese_resnet_model.pth" # path to save the trained model weights.

MODEL_CONFIG: Dict[str, Any] = {
    "embedding_dim": 128,
    "use_pretrained": True,
}


TRAINING_CONFIG: Dict[str, Any] = {
    "num_epochs": 30,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "weight_decay": 1e-5,
    "triplet_margin": 1.0,
}


DATA_TRANSFORMS: Dict[str, Any] = {
    "image_size": (100, 100),  
    "mean": [0.5],             
    "std": [0.5],              
}


EVAL_CONFIG: Dict[str, Any] = {

    "batch_size": 64,
}


LOGGING_CONFIG: Dict[str, Any] = {
    
    "log_dir": "runs/siamese_experiment",
    "enable_logging": True,
}
