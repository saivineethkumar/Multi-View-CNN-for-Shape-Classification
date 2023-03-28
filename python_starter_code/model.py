import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        """
        
        DEFINE YOUR NETWORK HERE
        
        """
        # input image size = 1x112x112

        # convolution layer 1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=0, bias=False)
        nn.init.kaiming_uniform_(self.conv1.weight)
        # layer normalization layer 1
        self.norm1 = nn.LayerNorm([8, 106, 106])
        # leaky ReLU layer 1
        self.relu1 = nn.LeakyReLU(0.01)
        # max pooling layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        
        # depthwise convolution layer
        self.depthwise_conv = nn.Conv2d(8, 8, kernel_size=7, stride=2, padding=0, groups=8, bias=False)
        nn.init.kaiming_uniform_(self.depthwise_conv.weight)
        # layer normalization layer 2
        self.norm2 = nn.LayerNorm([8, 24, 24])
        # leaky ReLU layer 2
        self.relu2 = nn.LeakyReLU(0.01)
        # max pooling layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # pointwise convolution layer
        self.pointwise_conv = nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.pointwise_conv.weight)
        nn.init.zeros_(self.pointwise_conv.bias)


        # depthwise convolution layer
        self.depthwise_conv2 = nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=0, groups=16, bias=False)
        nn.init.kaiming_uniform_(self.depthwise_conv2.weight)
        # layer normalization layer 3
        self.norm3 = nn.LayerNorm([16, 6, 6])
        # leaky ReLU layer 3
        self.relu3 = nn.LeakyReLU(0.01)
        # max pooling layer 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # pointwise convolution layer
        self.pointwise_conv2 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.pointwise_conv2.weight)
        nn.init.zeros_(self.pointwise_conv2.bias)


        # fully connected layer
        self.fc = nn.Conv2d(32, num_classes, kernel_size=3, bias=True)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


    def forward(self, x):
        """

        DEFINE YOUR FORWARD PASS HERE

        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.depthwise_conv(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.pointwise_conv(x)

        x = self.depthwise_conv2(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.pointwise_conv2(x)
        
        x = self.fc(x)
        x = x.squeeze()
        
        return x
