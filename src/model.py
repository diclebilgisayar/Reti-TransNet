# ==========================================
# 4. src/model.py (Mimari)
# ==========================================
with open(f"{Reti-TransNet}/src/model.py", "w") as f:
    f.write("""
import torch
import torch.nn as nn
import timm

class RetiTransNet(nn.Module):
    def __init__(self, num_classes=5):
        super(RetiTransNet, self).__init__()
        
        # 1. Local Branch (CNN) - EfficientNet-B0
        # num_classes=0: Sadece öznitelik vektörü al (Sınıflandırma katmanı yok)
        self.cnn = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.d_c = self.cnn.num_features  # 1280
        
        # 2. Global Branch (Transformer) - Swin Tiny
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        self.d_t = self.swin.num_features # 768
        
        # 3. Feature Alignment (Projeksiyon)
        # İki kolu da ortak boyuta (512) indirgiyoruz
        self.common_dim = 512
        self.proj_cnn = nn.Linear(self.d_c, self.common_dim)
        self.proj_swin = nn.Linear(self.d_t, self.common_dim)
        
        # 4. Adaptive Gated Fusion Mechanism
        # alpha = sigmoid(W * [V_cnn || V_swin] + b)
        self.gate = nn.Sequential(
            nn.Linear(self.common_dim * 2, self.common_dim),
            nn.ReLU(),
            nn.Linear(self.common_dim, self.common_dim),
            nn.Sigmoid()
        )
        
        # 5. Classifier
        self.classifier = nn.Linear(self.common_dim, num_classes)

    def forward(self, x):
        # Feature Extraction
        v_cnn = self.cnn(x)
        v_swin = self.swin(x)
        
        # Alignment
        v_cnn_aligned = self.proj_cnn(v_cnn)
        v_swin_aligned = self.proj_swin(v_swin)
        
        # Gating
        combined = torch.cat([v_cnn_aligned, v_swin_aligned], dim=1)
        alpha = self.gate(combined)
        
        # Fusion (Weighted Sum)
        z_fused = (alpha * v_cnn_aligned) + ((1 - alpha) * v_swin_aligned)
        
        return self.classifier(z_fused)
""")