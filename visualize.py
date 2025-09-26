"""

Author: Zaryab Rahman
Data: 23/9/25


"""

"""
visualization utilities for mmodel analysis

this module provides a set of functions for creating visual representations of the
model's training progress and performance. Visualizations are a crucial tool for
understanding machine learning models, helping to diagnose problems like overfitting,
and to intuitively grasp the model's capabilities.

The functions provided here are:
  `plot_loss_history`: Generates a plot of training and validation loss over epochs.
      This is essential for monitoring the
      learning process and identifying issues like
      overfitting (when validation loss
      starts increasing while training loss decreases)
      or underfitting (when both 
      losses remain high).
  `plot_roc_curve`: Generates a plot of the Receiver 
      Operating Characteristic (ROC) curve.
      This plot is a standard way to visualize the 
      performance of a binary classifier,
      showing the trade-off between the true positive
      rate and the false positive rate.
"""

from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_history(loss_history: Dict[str, List[float]], save_path: str = "loss_history.png") -> None:
    """
    
    plots the training and validation loss curves over epochs.

    this function takes a dictionary containing lists of training and validation losses
    and creates a line plot to visualize how these metrics changed during training.
    The resulting plot is displayed and saved to a file.

    args:
        loss_history (Dict[str, List[float]]): A dictionary with keys 'train_loss' and
                                               'val_loss', each mapped
                                                 to a list of
                                               loss values for each epoch.
        save_path (str, optional): The file path where the 
                                    plot image will be saved.
                                   Defaults to 
                                   "loss_history.png".

                                   
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    epochs = range(1, len(loss_history['train_loss']) + 1)

    ax.plot(epochs, loss_history['train_loss'], 'o-', color='royalblue', label='Training Loss')
    ax.plot(epochs, loss_history['val_loss'], 's-', color='darkorange', label='Validation Loss')

    ax.set_title('Model Training and Validation Loss', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Triplet Loss', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Loss history plot saved to {save_path}")
    plt.show()


def plot_roc_curve(eval_results: Dict, save_path: str = "roc_curve.png") -> None:
    """
    Plots ROC curve.

    this function uses the evaluation results (specifically the false positive rates,
    true positive rates, and AUC score) to generate a standard ROC curve plot. This
    visualization is critical for understanding the trade-offs of the classifier.

    args:
        eval_results (Dict): The dictionary returned by the
                              ModelEvaluator's `evaluate` method.
                             It must contain 'fpr', 'tpr', 
                             and 'roc_auc' keys.
        save_path (str, optional): The file path where 
                              the plot image will be saved.
                              Defaults to "roc_curve.png".
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    fpr = eval_results['fpr']
    tpr = eval_results['tpr']
    roc_auc = eval_results['roc_auc']

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.5)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"ROC curve plot saved to {save_path}")
    plt.show()
