"""The main Water Class for building a water network CNN."""

from torch import nn

class Water(nn.Module):
    """
    A module for building Convolutional Neural Networks.

    With automatic configuration of padding and stride.
    """
    def __init__(
        self,
        image_size=None,
        previous=None,
        out_channels=1,
        kernel=(3, 3),
        reduction=1,
        amplify=2,
        use_maxpool=False,
        use_avgpool=False,
        use_normalization=False,
    ):
        """
        Initialize the Water module.

        Arguments
        ---------
            image_size : tuple
                The target image specification (width, height, channels).
            previous : Water
                The previous Water module in the network.
            out_channels : int
                The number of output channels for the Conv2D layer.
            kernel : tuple
                The kernel size for the Conv2D layer.
            reduction : int
                The desired reduction factor for the output image size.
            amplify : int
                The amplification factor for MaxPool2D and AvgPool2D layers.
            use_maxpool : bool
                Whether to include a MaxPool2D layer.
            use_avgpool : bool
                Whether to include an AvgPool2D layer.
            use_normalization : bool
                Whether to include a BatchNorm2D layer.
        """
        super().__init__()

        if use_maxpool and use_avgpool:
            raise ValueError("MaxPool and AvgPool cannot co-exist.")

        self.previous = previous
        self.use_maxpool = use_maxpool
        self.use_avgpool = use_avgpool
        self.use_normalization = use_normalization

        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        else:
            self.kernel = kernel

        if self.previous is not None:
            in_channels = previous.out_channels
            self.image_size = previous.output_image_size
        else:
            in_channels = image_size[2]
            self.image_size = image_size[:-1]

        self.out_channels = out_channels
        padding, stride = self.calculate_padding_and_stride(
            self.image_size, kernel, reduction
        )

        self.conv = nn.Conv2d(
            in_channels, self.out_channels, kernel, stride, padding)
        self.output_image_size = self.calculate_output_image_size(
            self.image_size, kernel, stride, padding
        )

        if use_normalization:
            self.norm = nn.BatchNorm2d(self.out_channels)
        if use_maxpool:
            self.pool = nn.MaxPool2d(amplify, amplify)
            self.output_image_size = (
                self.output_image_size[0] // amplify,
                self.output_image_size[1] // amplify,
            )
        if use_avgpool:
            self.pool = nn.AvgPool2d(amplify, amplify)
            self.output_image_size = (
                self.output_image_size[0] // amplify,
                self.output_image_size[1] // amplify,
            )

    def forward(self, x):
        """
        Define the forward pass of the Water module.

        Arguments
        ---------
            x : torch.Tensor
                The input tensor.

        Returns
        -------
            : torch.Tensor
                The output tensor after passing through the Water module.
        """
        x = self.conv(x)
        if self.use_normalization:
            x = self.norm(x)
        if self.use_maxpool or self.use_avgpool:
            x = self.pool(x)
        return x

    def calculate_padding_and_stride(self, image_size, kernel, reduction):
        """
        Calculate padding and stride values based on the specified
        reduction parameter.

        Arguments
        ---------
            image_size : tuple
                The input image size (height, width).
            kernel : Union[int, tuple]
                The kernel size for the Conv2D layer.
            reduction : int
                The desired reduction factor for the output image size.

        Returns
        -------
            tuple: A tuple containing the calculated padding and stride values
                   (padding, stride).
        """
        padding_h = (reduction - 1) * image_size[0] - reduction + kernel[0]
        padding_w = (reduction - 1) * image_size[1] - reduction + kernel[1]
        padding = (padding_h // 2, padding_w // 2)
        stride = (reduction, reduction)
        return padding, stride


    def calculate_output_image_size(self, image_size, kernel, stride, padding):
        """
        Calculate the output image size after applying the Conv2D layer.

        Arguments
        ---------
            image_size : tuple
                The input image size (height, width).
            kernel : tuple
                The kernel size for the Conv2D layer.
            stride : tuple
                The stride for the Conv2D layer.
            padding : tuple
                The padding for the Conv2D layer.

        Returns
        -------
            tuple: A tuple containing the output image size (height, width).
        """
        output_height = ((image_size[0] + 2 * padding[0] - kernel[0]) \
            // stride[0]) + 1
        output_width = ((image_size[1] + 2 * padding[1] - kernel[1]) \
            // stride[1]) + 1
        return (output_height, output_width)

    def get_linear_size(self):
        """
        Calculate the size of the tensor after flattening the output
        of the current layer.

        Returns
        -------
            : int
                The total size of the flattened tensor.
        """
        return self.out_channels * self.output_image_size[0] * self.output_image_size[1]
