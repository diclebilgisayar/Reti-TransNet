import torch
import torch.nn as nn
import timm

class RetiTransNet(nn.Module):
    def __init__(self, num_classes=5):
        super(RetiTransNet, self).__init__()
        
        self.cnn = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.d_c = self.cnn.num_features
        
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.d_t = self.swin.num_features
        
        self.common_dim = 512
        self.proj_cnn = nn.Linear(self.d_c, self.common_dim)
        self.proj_swin = nn.Linear(self.d_t, self.common_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(self.common_dim * 2, self.common_dim),
            nn.ReLU(),
            nn.Linear(self.common_dim, self.common_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Linear(self.common_dim, num_classes)

    def forward(self, x):
        f_cnn = self.proj_cnn(self.cnn(x))
        f_swin = self.proj_swin(self.swin(x))
        
        combined = torch.cat([f_cnn, f_swin], dim=1)
        alpha = self.gate(combined)
        z = (alpha * f_cnn) + ((1 - alpha) * f_swin)
        
        return self.classifier(z)
