
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights, efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
from PIL import Image
from torchvision import transforms

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

        fused_input = torch.cat([weighted_global, weighted_local], dim=1)
        fused = self.fusion(fused_input)

        return fused, attn_weights

# ================== Modèle Complet avec Attention ==================
class AdvancedLocalGlobalNet(nn.Module):
    """Modèle final avec toutes les optimisations"""
    def __init__(self, num_classes, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # Branche globale : ViT
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vit_dim = vit.heads.head.in_features
        vit.heads.head = nn.Identity()
        self.vit = vit

        # Branche locale : EfficientNet
        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        eff_dim = eff.classifier[1].in_features
        eff.classifier = nn.Identity()
        self.eff = eff

        if use_attention:
            # Attention sur patches
            self.patch_attention = PatchAttention(eff_dim)
            # Fusion avec attention
            self.attention_fusion = AttentionFusion(vit_dim, eff_dim, hidden_dim=384)
            classifier_input = 384
        else:
            classifier_input = vit_dim + eff_dim

        # Classifier avec forte régularisation
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Pour Grad-CAM
        self.global_features = None
        self.local_features = None
        self.fusion_attention_weights = None
        self.patch_attention_weights = None

    def forward(self, img_global, patches_local):
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

# ================== Transformations ==================
def get_transforms():
    val_transform = transforms.Compose([
        transforms.Resize((Config.img_size_global, Config.img_size_global)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    local_transform = transforms.Compose([
        transforms.Resize((Config.img_size_local, Config.img_size_local)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return val_transform, local_transform

# ================== Patch Extraction (Simplified) ==================
def extract_random_patches(pil_img, num_patches=4, patch_size=96):
    w, h = pil_img.size
    ps = patch_size
    patches = []

    if w < ps or h < ps:
        pil_img = pil_img.resize((max(w, ps), max(h, ps)))
        w, h = pil_img.size

    for _ in range(num_patches):
        x = np.random.randint(0, max(1, w - ps + 1))
        y = np.random.randint(0, max(1, h - ps + 1))
        patch = pil_img.crop((x, y, min(x + ps, w), min(y + ps, h)))
        if patch.size != (ps, ps):
            patch = patch.resize((ps, ps))
        patches.append(patch)
    return patches
