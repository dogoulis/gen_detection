import torch 
import torch.nn as nn
import timm


# RRG class:

class RRG(nn.Module):
    def __init__(self):
        super(RRG, self).__init__()

        # avgpool, maxpool:
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        # activation functions:
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # starter convolutions:
        self.conv_3x3_in = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1, bias=False) # initial convolution
        self.conv_3x3_sec = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1, bias=False) # secondary convolution
        
        # spatial convolution:
        self.conv_3x3_sp = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False) # spatial convolution
        
        # channel convolution:
        self.conv_1x1_ch_in = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0, dilation=1, bias=False) # initial channel convolution
        self.conv_1x1_ch_sec = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=1, bias=False) # secondary channel convolution

        # final convolution:
        self.conv2d_1x1 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=0, dilation=1, bias=False) # final convolution

    def forward(self, x):
       # get a clone of the input x:
        x_input = x.clone()
       # intial convolutions and relu activations:
        x = self.conv_3x3_in(x)
        x = self.relu(x)
        x = self.conv_3x3_sec(x)

        # ---------- SPATIAL ATTENTION ----------
        # spatial-wise attention:
        # first calculate the avg
        x_spatial_avg = torch.mean(x, dim=1, keepdim=True)
        # then calculate the max
        x_spatial_max, _ = torch.max(x, dim=1, keepdim=True)

        #concatenate those feature maps:

        x_spatial = torch.cat([x_spatial_avg, x_spatial_max], dim=1) # shape: (batch_size, 2, height, width)

        # spatial convolution:
        x_spatial = self.conv_3x3_sp(x_spatial)
        x_spatial = self.sigmoid(x_spatial)

        # spatial feature map:
        x_spatial = x_spatial * x # shape: (batch_size, 3, height, width)

        # ---------- CHANNEL ATTENTION ----------
        # channel-wise attention:
        x_channel = self.avgpool(x) # shape: (batch_size, 3, 1, 1)
        x_channel = self.conv_1x1_ch_in(x_channel) # shape: (batch_size, 16, 1, 1)
        x_channel = self.relu(x_channel)
        x_channel = self.conv_1x1_ch_sec(x_channel) 
        x_channel = self.sigmoid(x_channel) 


        # channel feature map:
        x_channel = x_channel * x

        # ---------- CONCATENATE ----------
        # final feature map:
        x_conc = torch.cat([x_spatial, x_channel], dim=1) 
        

        # reduce channels dimension
        x_conc = self.conv2d_1x1(x_conc) # it has to be in the same shape as the input x

        # ---------- RESIDUAL CONNECTION ----------
        # residual connection:
        x_added = x_input + x_conc
        
        return x_added

class DNN(nn.Module):
    def __init__(self,):
        super(DNN, self).__init__()

        # first convolution:
        self.conv2d_3x3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1, bias=False) # initial convolution

        # final convolution:
        self.conv2d_3x3_sec = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1, bias=False) # final convolution

        # First RRG block:
        self.RRG_1 = RRG()

        # Second RRG block:
        self.RRG_2 = RRG()

        # Third RRG block:
        self.RRG_3 = RRG()

        # Fourth RRG block:
        self.RRG_4 = RRG()

        # Fifth RRG block:
        self.RRG_5 = RRG()

            
    def forward(self, x):
        # get a copy of the input x:
        x_input = x.clone()

        # ---------- ADD THE NOISE ----------
        # first convolution:
        x = self.conv2d_3x3(x)

        # pass through RRG blocks:
        x = self.RRG_1(x)
        x = self.RRG_2(x)
        x = self.RRG_3(x)
        x = self.RRG_4(x)
        x = self.RRG_5(x)

        # final convolution:
        x = self.conv2d_3x3_sec(x)

        # ---------- RESIDUAL CONNECTION ----------
        x_noise = x - x_input
        return x_noise, x