import os
import sys

def append_parent_dir(currentdir):
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)
    return parentdir

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = append_parent_dir(currentdir)
append_parent_dir(parentdir) 

import torch
from torch import nn


class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, loss_fn=nn.CrossEntropyLoss(), epochs=5, print_every=5):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.print_every = print_every
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_step(self, dataloader):
        self.model.train()
        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            y_pred = self.model(X)

            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        train_loss /= len(dataloader)
        train_acc /= len(dataloader)
        return train_loss, train_acc

    def test_step(self, dataloader):
        self.model.eval()
        test_loss, test_acc = 0, 0

        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)

                test_pred_logits = self.model(X)

                loss = self.loss_fn(test_pred_logits, y)
                test_loss += loss.item()

                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc

    def train(self):
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_step(self.train_dataloader)
            test_loss, test_acc = self.test_step(self.test_dataloader)

            if (epoch + 1) % self.print_every == 0 or epoch == 0:
                print(
                    f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc:.4f} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_acc: {test_acc:.4f}"
                )

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        return results
