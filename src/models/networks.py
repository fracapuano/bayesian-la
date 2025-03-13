import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic ResNet block with optional skip connection."""
    
    def __init__(self, in_channels, out_channels, stride=1, use_skip=True):
        super(BasicBlock, self).__init__()
        self.use_skip = use_skip
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply skip connection if specified
        if self.use_skip:
            out += self.shortcut(x)
            
        out = F.relu(out)
        return out

class ResNet5(nn.Module):
    """A minimal ResNet architecture to minimize computational overhead.
    
    This is essentially a small ResNet with 5 layers (2 residual blocks + input/output layers).
    Can be configured with or without skip connections.
    """
    
    def __init__(self, num_classes=10, use_skip=True):
        super(ResNet5, self).__init__()
        self.use_skip = use_skip
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Residual blocks
        self.layer1 = BasicBlock(16, 32, stride=2, use_skip=use_skip)
        self.layer2 = BasicBlock(32, 64, stride=2, use_skip=use_skip)
        
        # Final classification layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    @property
    def name(self):
        return f"ResNet5{'_Skip' if self.use_skip else '_NoSkip'}" 