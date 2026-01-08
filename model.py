import torch
import torch.nn as nn
import timm

class AdaptiveGatedFusion(nn.Module):
    """
    Implements the Dynamic Feature Alignment and Gating mechanism described in the paper.
    Formula: Z = alpha * V_cnn + (1 - alpha) * V_trans
    """
    def __init__(self, dim):
        super(AdaptiveGatedFusion, self).__init__()
        
        # Gating Network (The "Dynamic Spotlight")
        # Input: Concat[V_cnn, V_trans] -> Output: Gating Vector (alpha)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid() # Ensures values are between 0 and 1
        )
    
    def forward(self, v_cnn, v_trans):
        # 1. Concatenate aligned features
        combined = torch.cat([v_cnn, v_trans], dim=1)
        
        # 2. Compute Gating Vector
        alpha = self.gate(combined)
        
        # 3. Weighted Fusion (Element-wise)
        z_fused = (alpha * v_cnn) + ((1 - alpha) * v_trans)
        
        return z_fused

class RetiTransNet(nn.Module):
    def __init__(self, num_classes=5):
        super(RetiTransNet, self).__init__()
        
        # --- Local Branch (CNN) ---
        # EfficientNet-B0 (ImageNet Pretrained)
        # num_classes=0 returns the feature vector before classification
        self.cnn = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.d_c = self.cnn.num_features  # 1280
        
        # --- Global Branch (Transformer) ---
        # Swin Transformer Tiny
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.d_t = self.swin.num_features # 768
        
        # --- Feature Alignment (Projections) ---
        # Project both to a common latent dimension (e.g., 512)
        self.common_dim = 512
        self.proj_cnn = nn.Linear(self.d_c, self.common_dim)
        self.proj_swin = nn.Linear(self.d_t, self.common_dim)
        
        # --- Fusion Module ---
        self.fusion = AdaptiveGatedFusion(self.common_dim)
        
        # --- Final Classifier ---
        self.classifier = nn.Linear(self.common_dim, num_classes)

    def forward(self, x):
        # 1. Feature Extraction
        v_cnn = self.cnn(x)     # (Batch, 1280)
        v_swin = self.swin(x)   # (Batch, 768)
        
        # 2. Alignment
        v_cnn_aligned = self.proj_cnn(v_cnn)   # (Batch, 512)
        v_swin_aligned = self.proj_swin(v_swin) # (Batch, 512)
        
        # 3. Adaptive Fusion
        z_fused = self.fusion(v_cnn_aligned, v_swin_aligned)
        
        # 4. Classification
        out = self.classifier(z_fused)
        return out
