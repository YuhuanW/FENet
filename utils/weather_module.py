import torch.nn as nn
import torch.nn.functional as F

class WeatherModule(nn.Module):
    """
    Lightweight weather condition estimation module.
    Designed to be simple and effective for estimating weather impact on detection.
    """
    def __init__(self):
        super(WeatherModule, self).__init__()
        # Compact encoder: 2 conv layers with reasonable capacity
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        # Simple classifier: single FC layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output range [0, 1]
        )

    def forward(self, x):
        # Optionally downsample for efficiency (uncomment if needed)
        # For 640x640 input, downsampling to 160x160 can significantly reduce computation
        # x = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        
        x = self.encoder(x)
        return self.classifier(x)  # shape: [batch, 1]
