import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class StochasticBottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, survival_prob=1.0):
        super(StochasticBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.survival_prob = survival_prob
        
    def forward(self, x):
        if self.training and random.random() > self.survival_prob:
            residual = torch.zeros(x.size(0), self.conv2.out_channels, x.size(2), x.size(3), 
                                  device=x.device)
            return torch.cat([residual, x], 1)
        
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        
        if self.training:
            out = out / self.survival_prob
            
        out = torch.cat([out, x], 1)
        return out

class TransitionD(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TransitionD, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # ResNet-D style: first pool then convolve
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        
    def forward(self, x):
        out = self.relu(self.bn(x))
        out = self.pool(out)  # Pool first (ResNet-D style)
        out = self.conv(out)  # Then convolve
        return out

class StochasticDenseNetD(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, 
                 stochastic_depth_rate=0.2):
        super(StochasticDenseNetD, self).__init__()
        self.growth_rate = growth_rate
        num_planes = 2*growth_rate
        
        # ResNet-D style stem: three 3×3 convolutions
        self.conv1 = nn.Conv2d(3, num_planes//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_planes//2)
        self.conv2 = nn.Conv2d(num_planes//2, num_planes//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_planes//2)
        self.conv3 = nn.Conv2d(num_planes//2, num_planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Calculate survival probabilities for each block
        total_blocks = sum(nblocks)
        
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], 
                                             stochastic_depth_rate, 0, total_blocks)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = TransitionD(num_planes, out_planes)  # Using ResNet-D style transition
        num_planes = out_planes
        
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], 
                                             stochastic_depth_rate, nblocks[0], total_blocks)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = TransitionD(num_planes, out_planes)  # Using ResNet-D style transition
        num_planes = out_planes
        
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], 
                                             stochastic_depth_rate, nblocks[0]+nblocks[1], total_blocks)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = TransitionD(num_planes, out_planes)  # Using ResNet-D style transition
        num_planes = out_planes
        
        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], 
                                             stochastic_depth_rate, nblocks[0]+nblocks[1]+nblocks[2], total_blocks)
        num_planes += nblocks[3]*growth_rate
        
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)
        
    def _make_dense_layers(self, block, in_planes, nblock, stochastic_depth_rate, start_block, total_blocks):
        layers = []
        for i in range(nblock):
            layer_idx = start_block + i
            survival_prob = 1.0 - (layer_idx / float(total_blocks)) * stochastic_depth_rate
            
            layers.append(block(in_planes, self.growth_rate, survival_prob))
            in_planes += self.growth_rate
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # ResNet-D style stem forward pass
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def densenet_cifar_stochastic_d(stochastic_depth_rate=0.2):
    return StochasticDenseNetD(StochasticBottleneck, [6, 12, 24, 16], 
                              growth_rate=27, stochastic_depth_rate=stochastic_depth_rate)
