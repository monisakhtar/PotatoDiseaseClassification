import torch
from torch import nn 
torch.manual_seed(42)
torch.cuda.manual_seed(42)
class CNN_model(nn.Module):
  def __init__(self, input_shape:int, hidden_units: int, output_shape : int):
    super().__init__()
    self.cnn_layers = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride = 1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride =2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units *128 * 128,
                  out_features=output_shape)
    )
  def forward(self, x: torch.Tensor):
    x = self.cnn_layers(x)
    # print(x.shape)
    x = self.classifier(x)
    # print(x.shape)
    return x