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
        # 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2,2)
        # Batch normalization 
        self.bn1 = nn.BatchNorm2d(32)
        
        
        # second conv layer: 32 inputs, 64 outputs, 4x4 conv
        ## output size = (W-F)/S +1 = (110-4)/1 +1 = 107
        # the output tensor will have dimensions: (64, 107, 107)
        # after another pool layer this becomes (64, 53, 53); 53.5 rounded down
        self.conv2 = nn.Conv2d(32, 64, 4)
        # Batch normalization 
        self.bn2 = nn.BatchNorm2d(64)
        
        # third conv layer: 64 inputs, 128 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (53-3)/1 +1 = 51
        # the output tensor will have dimensions: (128, 51, 51)
        # after another pool layer this becomes (128, 25, 25); 25.5 rounded down
        self.conv3 = nn.Conv2d(64, 128, 3)
        # Batch normalization 
        self.bn3 = nn.BatchNorm2d(128)
        
        # fourth conv layer: 128 inputs, 256 outputs, 2x2 conv
        ## output size = (W-F)/S +1 = (25-2)/1 +1 = 24
        # the output tensor will have dimensions: (256, 24, 24)
        # after another pool layer this becomes (256, 12, 12)
        self.conv4 = nn.Conv2d(128, 256, 2)
        # Dropout with probability of 0.2
        self.drop1 = nn.Dropout2d(0.2)
        
        # fourth conv layer: 256 inputs, 512 outputs, 1x1 conv
        ## output size = (W-F)/S +1 = (12-1)/1 +1 = 12
        # the output tensor will have dimensions: (512, 12, 12)
        # after another pool layer this becomes (512, 6, 6)
        self.conv5 = nn.Conv2d(256, 512, 1)
        # 256 outputs from the conv layer * the 6*6 filtered/pooled map size
        self.fc1 = nn.Linear(512*6*6, 1000)
        # dropout with p=0.4
        self.drop2 = nn.Dropout(p=0.4)
        # finally, create 136 output channels (for the 68 keypoints in x,y coordinates)
        self.fc2 = nn.Linear(1000, 136)
        
        

        
    def forward(self, x):
        
        # 5 conv/relu + pool layers + dropout
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv5(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        
        # final output
        return x
