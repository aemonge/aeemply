# Water

A Python library for building Convolutional Neural Networks (CNNs) with a focus
on ease of use and automatic configuration of padding and stride.

## Features

- Automatically adjust padding and stride based on the specified `reduction`
  parameter.
- Provides options to apply `BatchNorm2d`, `MaxPool2d`, and `AvgPool2d`.
- Calculates the last size of the image for flattening and setting up a
  `Linear` layer.
- Estimates the total memory that the network would require.
- Compliant with PEP8 guidelines.
- Supports Python 3.7.6.

## Acceptance Criteria

- Initialize with the target image specification (width, height, channels).
- Generates the parameters of the `Conv2D` based on the previous connection,
  to allow connection between two or more layers, without mind calculations.
- Use the `kernel`, `padding` and `string` parameters of the `Conv2D`.
- Adjust padding and stride automatically based on the specified `reduction`
  parameter.
- Provides a function that calculates the last size of the image.
  Meant for a last Flatting layer connection, with out mind calculations.
- Provides a method that estimates the total memory that the network would
  require.
- Includes an option to apply `BatchNorm2d`.
- Includes an option to apply `MaxPool2d` with a customizable `amplify`
  parameter.
- Includes an option to apply `AvgPool2d` with a customizable `amplify`
  parameter.
- `MaxPool2d` and `AvgPool2d` cannot co-exist.
  An error would raises if attempted to.

## Usage

```python
from water import Water
import torch.nn as nn

water_layers = []
water_layers.append(Water(image_size=(224, 256, 3),
  out_channels=3, kernel=(2,2))
)
water_layers.append(Water(previous=water_layers[0],
  out_channels=16, kernel=3, reduction=2,
  use_maxpool=True
))
water_layers.append(Water(previous=water_layers[1],
  out_channels=32, kernel=2,
  use_maxpool=True, use_normalization=True
))
water_layers.append(Water(previous=water_layers[2],
  out_channels=64, kernel=2, amplify=3,
  use_maxpool=True
))
water_layers.append(Water(previous=water_layers[3],
  out_channels=16, kernel=2,
  use_maxpool=True
))
water_layers.append(Water(previous=water_layers[4],
  out_channels=8, kernel=2, reduction=3,
  use_avgpool=True, use_normalization=True
))
linear_size = water_layers[5].get_linear_size()

model = nn.Sequential(
  water_layers[0],
  nn.ReLU(),

  water_layers[1],
  nn.Dropout(0.5),
  nn.Tanh(inplace=True),

  water_layers[2],
  nn.Dropout(0.5),
  nn.Tanh(inplace=True),

  water_layers[3],
  nn.Dropout(0.5),
  nn.Tanh(inplace=True),

  water_layers[4],
  nn.Dropout(0.5),
  nn.Tanh(inplace=True),

  water_layers[5],
  nn.ReLU(inplace=True),

  nn.Linear(in_features=linear_size, out_features=96)
  nn.ReLU(inplace=True),
  nn.Linear(in_features=96, out_features=50)
  nn.LogSoftmax(dim=1)
)
```
