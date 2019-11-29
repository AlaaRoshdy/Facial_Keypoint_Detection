## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
      
        # 1 input image channel (grayscale), 32 output channels/feature maps
        # 4x4 square convolution kernel
        ## output size = (W-F)/S +1 = (96-4)/1 +1 = 93
        # the output Tensor for one image, will have the dimensions: (32, 93, 93)
        # after one pool layer, this becomes (32, 46, 46)
        self.conv1 = nn.Conv2d(1, 32, 4)
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2,2)
        # Batch normalization 
        self.bn1 = nn.BatchNorm2d(32)
        
        
        # second conv layer: 32 inputs, 64 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (46-3)/1 +1 = 44
        # the output tensor will have dimensions: (64, 44, 44)
        # after another pool layer this becomes (64, 22, 22)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Batch normalization 
        self.bn2 = nn.BatchNorm2d(64)
        
        # third conv layer: 64 inputs, 128 outputs, 2x2 conv
        ## output size = (W-F)/S +1 = (22-2)/1 +1 = 21 
        # the output tensor will have dimensions: (128, 21, 21)
        # after another pool layer this becomes (128, 10, 10)
        self.conv3 = nn.Conv2d(64, 128, 2)
        #dropout with p = 0.2
        self.drop1 = nn.Dropout2d(0.2)
        
        
        # 256 outputs from the conv layer * the 2*2 filtered/pooled map size
        self.fc1 = nn.Linear(128*10*10, 1000)
        # dropout with p=0.4
        self.drop2 = nn.Dropout(p=0.4)
        # finally, create 136 output channels (for the 68 keypoints in x,y coordinates)
        self.fc2 = nn.Linear(1000, 136)
        
        

        
    def forward(self, x):
        
        # 3 conv/relu + pool layers + dropout
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop1(x)
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        
        # final output
        return x
