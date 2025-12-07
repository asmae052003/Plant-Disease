
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights, efficientnet_b0, EfficientNet_B0_Weights
import numpy as np

from PIL import Image

# ================== Configuration ==================
class Config:
    # Images
    img_size_global = 224
    img_size_local = 96
    patch_size = 96
    num_patches = 4

    # Modèle
    embed_dim = 384
    dropout = 0.5

    # Entraînement
    lr = 5e-5
    weight_decay = 0.05
    batch_size = 8  # Réduire à 8 si problème mémoire
    epochs = 20

    # Régularisation
    use_mixup = True
    mixup_alpha = 0.2
    label_smoothing = 0.1
    early_stopping_patience = 7
    warmup_epochs = 3

    # Features avancées
    use_smart_patches = True  # Extraction intelligente
    use_attention = True       # Mécanismes d'attention

# ================== Mécanismes d'Attention ==================
class PatchAttention(nn.Module):
    """Attention pour pondérer l'importance des patches"""
    def __init__(self, patch_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(patch_dim, patch_dim // 2),
            nn.Tanh(),
            nn.Linear(patch_dim // 2, 1)
        )

    def forward(self, patch_features):
        attn_scores = self.attention(patch_features)
        attn_weights = F.softmax(attn_scores, dim=1)
        weighted_features = patch_features * attn_weights
        aggregated = weighted_features.sum(dim=1)
        return aggregated, attn_weights.squeeze(-1)


class AttentionFusion(nn.Module):
    """Fusion avec attention global vs local"""
    def __init__(self, global_dim, local_dim, hidden_dim=256):
        super().__init__()
        self.global_proj = nn.Linear(global_dim, hidden_dim)
        self.local_proj = nn.Linear(local_dim, hidden_dim)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, global_feat, local_feat):
        global_h = self.global_proj(global_feat)
        local_h = self.local_proj(local_feat)

        combined = torch.cat([global_h, local_h], dim=1)
        attn_weights = self.attention(combined)

        weighted_global = global_h * attn_weights[:, 0:1]
        weighted_local = local_h * attn_weights[:, 1:2]
        B, N, C, H, W = patches_local.shape

        # Global
        feat_global = self.vit(img_global)

        # Local
        patches_flat = patches_local.view(B * N, C, H, W)
        feat_local_all = self.eff(patches_flat)
        feat_local_all = feat_local_all.view(B, N, -1)

        if self.use_attention:
            # Attention sur patches
            feat_local, patch_attn = self.patch_attention(feat_local_all)
            # Fusion avec attention
            fused, fusion_attn = self.attention_fusion(feat_global, feat_local)

            self.global_features = feat_global
            self.local_features = feat_local
            self.fusion_attention_weights = fusion_attn
            self.patch_attention_weights = patch_attn
        else:
            feat_local = feat_local_all.mean(dim=1)
            fused = torch.cat([feat_global, feat_local], dim=1)
            fusion_attn = None

        logits = self.classifier(fused)
        return logits, fusion_attn

if __name__ == "__main__":
    try:
        # Try to load the model
        # We might need to map the class name if it was saved as '__main__.AdvancedLocalGlobalNet'
        # But since we are in __main__ here, it might work if we just load it.
        
        # Sometimes models are saved as dictionaries
        checkpoint = torch.load('best_model_final.pth', map_location='cpu')
        
        if isinstance(checkpoint, dict):
            print("Checkpoint is a dictionary with keys:", checkpoint.keys())
            if 'state_dict' in checkpoint:
                print("Found state_dict.")
            if 'class_names' in checkpoint:
                print("Found class_names:", checkpoint['class_names'])
            elif 'classes' in checkpoint:
                print("Found classes:", checkpoint['classes'])
        elif isinstance(checkpoint, nn.Module):
            print("Checkpoint is a full model object.")
            if hasattr(checkpoint, 'classes'):
                print("Model has 'classes' attribute:", checkpoint.classes)
            elif hasattr(checkpoint, 'class_names'):
                print("Model has 'class_names' attribute:", checkpoint.class_names)
            else:
                print("Model does not have explicit class attributes.")
                # Try to infer num_classes
                try:
                    print("Last layer:", checkpoint.classifier[-1])
                except:
                    pass
        else:
            print("Checkpoint is of type:", type(checkpoint))

    except Exception as e:
        print(f"Error loading model: {e}")
