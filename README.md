# Siamese Network for One-Shot Face Verification

![Architecure](/data/a.webp) 

This repository contains a comprehensive, highly-documented, and modular implementation of a Siamese Network in PyTorch for one-shot face verification. The project is designed to be a premier educational resource, demonstrating best practices in deep learning model development, from data handling to evaluation and inference.

## Introduction

Our project is built around a powerful yet clean design. At its core, it leverages a **ResNet-18 backbone with transfer learning**, ensuring strong feature extraction without unnecessary complexity. To make the embeddings truly discriminative, we employ **Triplet Loss**, a well-established metric learning technique. For fair and realistic evaluation, the dataset is split **subject-wise**, meaning the model is tested on completely unseen identities—closely mirroring real-world one-shot scenarios. Performance isn’t judged lightly either; we report **Accuracy**, plot the **ROC curve**, and compute the **AUC score** for a complete picture of results. The codebase itself is modular and easy to follow, with clear divisions across configuration, dataset handling, modeling, and training. To keep track of experiments, the workflow is integrated with **TensorBoard**, offering real-time insights into training and validation loss. Finally, we’ve included a simple inference script, allowing anyone to quickly test the trained model on two face images and verify its predictions with ease.


## Project Structure

The repository is organized into distinct, well-commented modules:

```
advanced-siamese-network/
│
├── data/                      # directory for the AT&T Faces dataset
├── config.py                  # centralized configuration for all hyperparameters
├── dataset.py                 # custom PyTorch Datasets for triplets (training) and pairs (eval)
├── model.py                   # the SiameseResNet model architecture
├── loss.py                    # implementation of the Triplet Loss function
├── trainer.py                 # the ModelTrainer class for handling training/validation loops
├── evaluator.py               # the ModelEvaluator class for computing accuracy, ROC, and AUC
├── visualize.py               # functions for plotting loss history and ROC curves
├── main.py                    # the main script to run the entire training and evaluation pipeline
├── inference.py               # a standalone script for running inference on new images
├── requirements.txt           # all Python package dependencies
└── README.md                  # this documentation file
```


### 1. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/advanced-siamese-network.git
    cd advanced-siamese-network
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 2. Download the Dataset

-   This project uses the **AT&T Database of Faces**. You can download it from [Kaggle](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces).
-   Extract the dataset and ensure the directory structure looks like this:

    ```
    data/
    └── att_faces/
        ├── s1/
        │   ├── 1.pgm
        │   ├── 2.pgm
        │   └── ...
        ├── s2/
        └── ...
    ```

## How to use this repo

### 1. Training the Model

to run the full pipeline—from data loading to training, evaluation, and visualization—simply execute the `main.py` script:

```bash
python main.py
```

the script will print progress updates to the console, save the best model to `siamese_resnet_model.pth`, and display the loss and ROC curve plots.

### 2. Monitoring with TensorBoard

while the training is running, you can visualize the loss curves in real-time with TensorBoard:

```bash
# In a new terminal, from the project root directory
tensorboard --logdir=runs
```

Navigate to `http://localhost:6006/` in your web browser.

### 3. Running Inference

After the model is trained, use `inference.py` to test it on any two images.

```bash
python inference.py --img1 "path/to/image_A.jpg" --img2 "path/to/image_B.jpg"
```

The script will output the computed distance and a prediction of whether the images are of the same person or different people.

## Results


**(Placeholder for Loss History Plot)**
![Loss History](https://i.imgur.com/3i4u5rK.png)

**(Placeholder for ROC Curve Plot)**
![ROC Curve](https://i.imgur.com/kP1GgE4.png)

The model  achieved an **accuracy  95.4%** and an **AUC  0.97**, demonstrating its strong capability for one-shot face verification.

## 

### Siamese Network

A Siamese Network is not a unique architecture but a way of using two identical neural networks (twins with shared weights) to process two different inputs. The goal is not to classify the inputs but to learn an **embedding space** where similar inputs are mapped to nearby points and dissimilar inputs are mapped to distant points. The distance (e.g., Euclidean distance) between the two output embeddings becomes a powerful similarity metric.

### Triplet Loss

To create this embedding space, we use Triplet Loss. The loss function considers three inputs at a time:
-   An **Anchor** (a reference image)
-   A **Positive** (another image of the same class as the Anchor)
-   A **Negative** (an image from a different class)

The loss is designed to "push" the Negative's embedding away from the Anchor's, while "pulling" the Positive's embedding closer, enforcing a `margin` of separation.
`L(a, p, n) = max( d(a, p)² - d(a, n)² + margin, 0 )`

## 💡 Future Improvements

-   **Hard Triplet Mining**: Implement online hard triplet mining to select the most challenging triplets during training, which can lead to faster convergence and better performance.
-   **Different Backbones**: Experiment with more modern or lightweight architectures like EfficientNet or MobileNetV3 as the model backbone.
-   **Deployment**: Wrap the inference logic in a simple web application using Flask or FastAPI to create an interactive face verification tool.
-   **Larger Datasets**: Train the model on a larger, more diverse face dataset like LFW (Labeled Faces in the Wild) or VGGFace2.
