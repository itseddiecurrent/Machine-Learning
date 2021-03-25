
#import matplotlib.pyplot as plt
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
#import torch.optim as optim



class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
                         
            i. A Conv2d with C input channels, C output channels, kernel size 3, stride 1, 
            padding 1, and no bias term.
            ii. A BatchNorm2d with C features.
            iii. A ReLU layer.
            iv. Another Conv2d with the same arguments as i above.
            v. Another BatchNorm2d with C features.
            Because 3 × 3 kernels and padding 1 are used, the convolutional layers do not 
            change the shape of each channel. Moreover, the number of channels are also kept 
            unchanged. Therefore f(x) does have the same shape as x.
        """
        super(Block, self).__init__()
        self.fc1 = nn.Conv2d(in_channels = num_channels, out_channels = num_channels, kernel_size = 3, 
                             stride = 1, padding = 1, bias = False)
        self.fc2 = nn.BatchNorm2d(num_channels)
        self.fc3 = nn.ReLU()
        self.fc4 = nn.Conv2d(in_channels = num_channels, out_channels = num_channels, kernel_size = 3, 
                             stride = 1, padding = 1, bias = False)
        self.fc5 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        xx = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc3(x+xx)
        return x


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
            i. A Conv2d with 1 input channel, C output channels, kernel size 3, 
            stride 2, padding 1, and no bias term.
            ii. A BatchNorm2d with C features.
            iii. A ReLU layer.
            iv. A MaxPool2d with kernel size 2.
            v. A Block with C channels.
            vi. An AdaptiveAvgPool2d which for each channel takes the average of all elements.
            vii. A Linear with C inputs and 10 outputs.
        """
        super(ResNet, self).__init__()
        self.fc1 = nn.Conv2d(in_channels = 1, out_channels = num_channels, kernel_size = 3,
                            stride = 2, padding = 1, bias = False)
        self.fc2 = nn.BatchNorm2d(num_channels)
        self.fc3 = nn.ReLU()
        self.fc4 = nn.MaxPool2d(kernel_size = 2)
        self.fc5 = Block(num_channels)
        self.fc6 = nn.AdaptiveAvgPool2d(1)
        self.fc7 = nn.Linear(num_channels, 10)
        
        
        

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        print("Initial ", x.shape)
        N, C, H, W = x.shape
        x = self.fc1(x)
        print(x.shape, " after fc1")
        x = self.fc2(x)
        print(x.shape, " after fc2")
        x = self.fc3(x)
        print(x.shape, " after fc3")
        x = self.fc4(x)
        print(x.shape, " after fc4")
        x = self.fc5(x)
        print(x.shape, " after fc5")
        x = self.fc6(x)
        print(x.shape, " after fc6")
        x = x.view(-1, x.shape[1])
        x = self.fc7(x)
        print(x.shape, " after fc7")
        print(x.shape)
        return x


def one_nearest_neighbor(X,Y,X_test):
    print("X shape is ", X.shape)
    print("Y shape is ", Y.shape)
    print("X test shape is ", X_test.shape)
    
    out = []
    
    for i in range(len(X_test)):
        mindist = distance(X[0], X_test[i])
        mindistIndex = 0
        for j in range(len(X)):
            dist = distance(X[j],X_test[i])
            if (mindist > dist):
                mindist = dist
                mindistIndex = j
        out.append(Y[mindistIndex])
        print(mindistIndex)
    return torch.from_numpy(np.asarray(out))
    

def distance(a,b):
    s = 0
    for i in range(len(a)):
        s += np.square(a[i]-b[i])
    return s
