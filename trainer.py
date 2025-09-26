"""
model training and validation 

this module defines the `ModelTrainer` class, which is responsible for  the
entire training and validation process of the Siamese Network. By encapsulating this logic
into a dedicated class, we separate the concerns of model training from the main application
flow (data loading, model definition, evaluation, etc.), leading to cleaner and more
maintainable code.

the trainer handles:
iterating through the dataset for a specified number of epochs.
executing the training step for each batch (forward pass, loss calculation, backpropagation).
executing the validation step for each batch to monitor performance on unseen data.
logging training and validation metrics (e.g., loss) to the console and TensorBoard.
implementing a basic early stopping mechanism by saving the model only when the
validation loss improves.
"""

import os
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class ModelTrainer:
    """
    a class to handle the training 
    and validation 
    loops for the Siamese Network.
    
    """

    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: optim.Optimizer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str,
                 config: Dict,
                 writer: SummaryWriter = None) -> None:
        """
        initializes the ModelTrainer.

        args:
            model (nn.Module): the Siamese Network model to be trained.
            loss_fn (nn.Module): the loss function to use (e.g., TripletLoss).
            optimizer (optim.Optimizer): the optimizer for updating model weights (e.g., Adam).
            train_loader (DataLoader): dataLoader for the training dataset.
            val_loader (DataLoader): dataLoader for the validation dataset.
            device (str): the device to run the training on ('cuda' or 'cpu').
            config (Dict): a dictionary containing training configurations like num_epochs, etc.
            writer (SummaryWriter, optional): a TensorBoard SummaryWriter for logging. Defaults to None.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.writer = writer
        self.best_val_loss = float('inf')  # Initialize with a very high value

    def _run_epoch(self, dataloader: DataLoader, is_training: bool) -> float:
        """
        
        executes a single epoch of training or validation.

        this is a helper function that abstracts the common logic of iterating over a
        dataloader. It handles the forward pass and loss calculation. If in training mode,
        it also performs backpropagation and updates the model weights.

        args:
            dataloader (DataLoader): the dataloader for the current epoch (either train or val).
            is_training (bool): a flag to indicate whether this is a training or validation epoch.
                                This controls whether to perform backpropagation and if the model
                                should be in `train()` or `eval()` mode.

        returns:
            float: The average loss for the epoch.
            
        """
        epoch_loss = 0.0
        # set to training or evaluation 
        # like Dropout and BatchNorm, 
        self.model.train() if is_training else self.model.eval()

        for anchor, positive, negative in dataloader:
            # move data to device the confg one 
            anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

            if is_training:
                
                #  zero the gradients: 
                #  essential before a new backward pass, otherwise
                #  gradients would accumulate from previous batches.
                self.optimizer.zero_grad()

                # compute embeddings for the triplet.
                emb_anchor, emb_positive, emb_negative = self.model(anchor, positive, negative)

                # compute the Triplet Loss based on the embeddings. f-p
                loss = self.loss_fn(emb_anchor, emb_positive, emb_negative)

                # compute gradients of the loss with respect to model parameters. b-p
                loss.backward()

                # update the model parameters using the computed gradients.
                self.optimizer.step()
            else:
                
                # `torch.no_grad()` disables gradient 
                #  calculation, which is not needed
                #  for validation. This reduces memory 
                #  consumption and speeds up computation.
                with torch.no_grad():
                    emb_anchor, emb_positive, emb_negative = self.model(anchor, positive, negative)
                    loss = self.loss_fn(emb_anchor, emb_positive, emb_negative)

            # accumulate the loss for the batch. 
            #   .item() extracts the scalar value.
            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def train(self) -> Dict[str, list]:
        """
        the main training loop  for the entire training process.

        this method iterates for a specified number of epochs. In each epoch, it runs
        the training and validation phases, logs the results, and saves the best
        performing model based on validation loss.

        returns:
            Dict[str, list]: A dictionary containing the history of training and validation
                             losses for each epoch. This can be used for plotting.
        """
        print("Starting model training...")
        train_loss_history = []
        val_loss_history = []

        for epoch in range(self.config['num_epochs']):
            train_loss = self._run_epoch(self.train_loader, is_training=True)
            train_loss_history.append(train_loss)

            val_loss = self._run_epoch(self.val_loader, is_training=False)
            val_loss_history.append(val_loss)

            print(f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Validation Loss: {val_loss:.4f}")

            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/validation', val_loss, epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config['model_save_path'])
                print(f"Validation loss improved. Model saved to {self.config['model_save_path']}")

        print("Training finished.")
        if self.writer:
            self.writer.close()

        return {"train_loss": train_loss_history, "val_loss": val_loss_history}
