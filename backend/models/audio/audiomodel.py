
import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


CFG = { # Audio processing 
    "sample_rate": 16000, "max_length_s": 4.0, "mel_n_mels": 80, "stft_n_fft": 2048, "stft_hop": 512, # Training 
    "batch_size": 16, "ssl_epochs": 5, "warmup_epochs": 3, "train_epochs": 30, "gradient_clip": 1.0, # Architecture 
    "embedding_dim": 256, "transformer_d_model": 512, "transformer_nhead": 8, "transformer_layers": 6, "artifact_base_channels": 64, # Regularization "
    "dropout": 0.15, "weight_decay": 1e-4, "label_smoothing": 0.1, # Advanced features 
    "use_mixup": True, "mixup_alpha": 0.4, "use_cutmix": True, "cutmix_alpha": 1.0, "use_focal_loss": True, "seed": 42, "num_workers": 1 
    }

# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# -------------------------
# Enhanced Structural Branch
# -------------------------
class EnhancedStructuralBranch(nn.Module):
    def __init__(self, n_mels=CFG["mel_n_mels"], d_model=CFG["transformer_d_model"], 
                 nhead=CFG["transformer_nhead"], num_layers=CFG["transformer_layers"]):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_mels, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(CFG["dropout"])
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=1000)
        
        # Transformer encoder layers with pre-norm
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=CFG["dropout"],
                activation="gelu",
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        # Multi-scale pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # SSL components
        self.mask_token = nn.Parameter(torch.randn(d_model))
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(d_model, n_mels)
        )
        
        # Classification head
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),  # *2 for avg + max pooling
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(d_model, CFG["embedding_dim"])
        )
    
    def forward(self, mel, mask_ratio=0.0, do_reconstruct=False):
        B, n_mels, T = mel.shape
        x = mel.transpose(1, 2)  # (B, T, n_mels)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # Apply masking for SSL
        if mask_ratio > 0:
            x = self.apply_masking(x, mask_ratio)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Multi-scale pooling
        x_transposed = x.transpose(1, 2)  # (B, d_model, T)
        avg_pooled = self.avg_pool(x_transposed).squeeze(-1)
        max_pooled = self.max_pool(x_transposed).squeeze(-1)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # Final embedding
        emb = self.classifier_head(pooled)
        
        if do_reconstruct:
            recon = self.reconstruction_head(x)
            return emb, recon
        
        return emb
    
    def apply_masking(self, x, mask_ratio):
        B, T, D = x.shape
        num_mask = max(1, int(T * mask_ratio))
        mask = torch.zeros((B, T), dtype=torch.bool, device=x.device)
        
        for i in range(B):
            idx = torch.randperm(T)[:num_mask]
            mask[i, idx] = True
        
        x[mask] = self.mask_token
        return x

# -------------------------
# Squeeze-and-Excitation Block
# -------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# -------------------------
# Enhanced Artifacts Branch
# -------------------------
class EnhancedArtifactsBranch(nn.Module):
    def __init__(self, in_channels=1, base_channels=CFG["artifact_base_channels"], 
                 embedding_dim=CFG["embedding_dim"]):
        super().__init__()
        
        # Multi-scale initial convolutions
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention for multi-scale fusion
        self.channel_attention = SEBlock(base_channels)
        
        # Residual blocks with SE attention
        self.res_block1 = self._make_res_block(base_channels, base_channels * 2)
        self.res_block2 = self._make_res_block(base_channels * 2, base_channels * 4)
        self.res_block3 = self._make_res_block(base_channels * 4, base_channels * 8)
        self.res_block4 = self._make_res_block(base_channels * 8, base_channels * 16)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(CFG["dropout"]),
            nn.Linear(base_channels * 16 * 2, 512),  # *2 for avg + max pooling
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(CFG["dropout"] * 0.5),
            nn.Linear(512, embedding_dim)
        )
    
    def _make_res_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            SEBlock(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, spec):
        B, F, T = spec.shape
        x = spec.unsqueeze(1)  # (B, 1, F, T)
        
        # Multi-scale initial features
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        feat7 = self.conv7x7(x)
        
        # Combine multi-scale features
        x = feat3 + feat5 + feat7
        x = self.channel_attention(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Global pooling
        avg_pooled = self.global_avg_pool(x).view(B, -1)
        max_pooled = self.global_max_pool(x).view(B, -1)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # Final embedding
        emb = self.classifier(pooled)
        return emb


# -------------------------
# Enhanced DBM-FC Model
# -------------------------
class EnhancedDBMFC(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.structural = EnhancedStructuralBranch()
        self.artifacts = EnhancedArtifactsBranch()
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=CFG["embedding_dim"],
            num_heads=8,
            dropout=CFG["dropout"],
            batch_first=True
        )
        
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(CFG["embedding_dim"] * 2, CFG["embedding_dim"]),
            nn.Sigmoid()
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(CFG["embedding_dim"] * 2, CFG["embedding_dim"] * 2),
            nn.LayerNorm(CFG["embedding_dim"] * 2),
            nn.GELU(),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(CFG["embedding_dim"] * 2, CFG["embedding_dim"])
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(CFG["embedding_dim"], CFG["embedding_dim"] // 2),
            nn.ReLU(),
            nn.Dropout(CFG["dropout"]),
            nn.Linear(CFG["embedding_dim"] // 2, num_classes)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(CFG["embedding_dim"], 1)
    
    def forward(self, mel, spec, mask_ratio=0.0, do_reconstruct=False):
        # Extract features
        if do_reconstruct:
            s_emb, recon = self.structural(mel, mask_ratio, do_reconstruct)
        else:
            s_emb = self.structural(mel, mask_ratio, do_reconstruct)
            recon = None
        
        a_emb = self.artifacts(spec)
        
        # Cross-modal attention
        s_emb_att, _ = self.cross_attention(
            s_emb.unsqueeze(1), a_emb.unsqueeze(1), a_emb.unsqueeze(1)
        )
        s_emb_att = s_emb_att.squeeze(1)
        
        # Gated fusion
        combined = torch.cat([s_emb_att, a_emb], dim=1)
        gate_weights = self.gate(combined)
        gated_s = s_emb_att * gate_weights
        gated_a = a_emb * (1 - gate_weights)
        
        # Final fusion
        fused_input = torch.cat([gated_s, gated_a], dim=1)
        emb = self.fusion(fused_input)
        
        # Classification and uncertainty
        logits = self.classifier(emb)
        uncertainty = self.uncertainty_head(emb)
        
        if do_reconstruct:
            return logits, emb, recon, uncertainty
        else:
            return logits, emb, uncertainty