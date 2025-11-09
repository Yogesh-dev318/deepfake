import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from preprocessing.audio_process import CFG
from models.audioModel.artifact_branch import ArtifactsBranch
from models.audioModel.structural_branch import StructuralBranch



# -------------------------
# Fusion + Full DBM-FC Model
# -------------------------
class DBMFC(nn.Module):
    def __init__(self, structural_module=None, artifacts_module=None, 
                 embedding_dim=CFG["embedding_dim"], num_classes=2):
        super().__init__()
        self.structural = structural_module if structural_module is not None else StructuralBranch()
        self.artifacts = artifacts_module if artifacts_module is not None else ArtifactsBranch()
        # fusion: concat structural + artifact embeddings -> project to final embedding
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim*2, embedding_dim*2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim*2, embedding_dim)
        )
        # classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, mel, spec, mask_ratio=0.0, do_reconstruct=False):
        if do_reconstruct:
            s_emb, recon = self.structural(mel, mask_ratio=mask_ratio, do_reconstruct=True)
        else:
            s_emb = self.structural(mel, mask_ratio=mask_ratio, do_reconstruct=False)
            recon = None
        a_emb = self.artifacts(spec)
        fused = torch.cat([s_emb, a_emb], dim=1)
        emb = self.fusion(fused)
        logits = self.classifier(emb)
        return logits, emb, recon

