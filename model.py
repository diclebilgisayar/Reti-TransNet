import torch
import torch.nn as nn
import timm

class RetiTransNet(nn.Module):
    def __init__(self, num_classes=5):
        super(RetiTransNet, self).__init__()
        
        # Local Branch (CNN)
        self.cnn = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.d_c = self.cnn.num_features
        
        # Global Branch (Transformer)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.d_t = self.swin.num_features
        
        # Alignment (Projection)
        self.common_dim = 512
        self.proj_cnn = nn.Linear(self.d_c, self.common_dim)
        self.proj_swin = nn.Linear(self.d_t, self.common_dim)
        
        # Adaptive Gated Fusion
        self.gate = nn.Sequential(
            nn.Linear(self.common_dim * 2, self.common_dim),
            nn.ReLU(),
            nn.Linear(self.common_dim, self.common_dim),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Linear(self.common_dim, num_classes)

    def forward(self, x):
        v_cnn = self.cnn(x)
        v_swin = self.swin(x)
        
        # Align
        v_cnn_aligned = self.proj_cnn(v_cnn)
        v_swin_aligned = self.proj_swin(v_swin)
        
        # Gate
        combined = torch.cat([v_cnn_aligned, v_swin_aligned], dim=1)
        alpha = self.gate(combined)
        
        # Fuse
        z_fused = (alpha * v_cnn_aligned) + ((1 - alpha) * v_swin_aligned)
        
        return self.classifier(z_fused)
