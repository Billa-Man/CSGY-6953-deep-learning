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
            # During training, if we drop this layer, we still need to maintain the same number of channels
            # Create a zero tensor with the same shape as the output would have been
            # This ensures channel count consistency throughout the network
            residual = torch.zeros(x.size(0), self.conv2.out_channels, x.size(2), x.size(3), 
                                  device=x.device)
            return torch.cat([residual, x], 1)
        
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        
        if self.training:
            # Scale during training by survival probability
            out = out / self.survival_prob
            
        out = torch.cat([out, x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class StochasticDenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, 
                 stochastic_depth_rate=0.2):
        super(StochasticDenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        
        # Calculate survival probabilities for each block
        # Linear decay of survival probability from 1 to 1-stochastic_depth_rate
        total_blocks = sum(nblocks)
        
        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], 
                                             stochastic_depth_rate, 0, total_blocks)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes
        
        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], 
                                             stochastic_depth_rate, nblocks[0], total_blocks)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes
        
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], 
                                             stochastic_depth_rate, nblocks[0]+nblocks[1], total_blocks)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes
        
        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], 
                                             stochastic_depth_rate, nblocks[0]+nblocks[1]+nblocks[2], total_blocks)
        num_planes += nblocks[3]*growth_rate
        
        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)
        
    def _make_dense_layers(self, block, in_planes, nblock, stochastic_depth_rate, start_block, total_blocks):
        layers = []
        for i in range(nblock):
            # Calculate survival probability based on layer position
            # Later layers have lower survival probability
            # Formula: 1 - (layer_idx / total_layers) * stochastic_depth_rate
            layer_idx = start_block + i
            survival_prob = 1.0 - (layer_idx / float(total_blocks)) * stochastic_depth_rate
            
            layers.append(block(in_planes, self.growth_rate, survival_prob))
            in_planes += self.growth_rate
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def densenet_cifar_stochastic(stochastic_depth_rate=0.2):
    return StochasticDenseNet(StochasticBottleneck, [6, 12, 24, 16], 
                              growth_rate=27, stochastic_depth_rate=stochastic_depth_rate)
