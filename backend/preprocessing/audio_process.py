import os, random
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torchaudio
import warnings
warnings.filterwarnings("ignore")


CFG = {
    "sample_rate": 16000,
    "max_length_s": 4.0,
    "mel_n_mels": 80,
    "stft_n_fft": 2048,
    "stft_hop": 512,
    "batch_size": 32,
    "ssl_epochs": 3,           # SSL pretraining epochs
    "train_epochs": 20,        # ← Updated to 20 epochs
    "embedding_dim": 128,
    "transformer_d_model": 256,
    "transformer_nhead": 4,
    "transformer_layers": 4,
    "artifact_base_channels": 32,
    "seed": 42,
    "num_workers": 4
}

class ASVSpoofDataset(Dataset):
    def __init__(self, audio_paths, labels, sample_rate=CFG["sample_rate"], max_length=CFG["max_length_s"],
                 augment=False):
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_length)
        self.augment = augment

        # Mel spectrogram for Structural branch
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=CFG["stft_hop"],
            n_mels=CFG["mel_n_mels"],
            f_min=0.0, f_max=sample_rate//2
        )
        # High-fidelity spectrogram for Artifact branch (power spectrogram)
        self.spec_transform = T.Spectrogram(
            n_fft=CFG["stft_n_fft"],
            hop_length=CFG["stft_hop"],
            power=2.0,
            normalized=False
        )
        # augment primitives
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    # def load_audio(self, path):
    #     waveform, sr = torchaudio.load(path)
    #     if sr != self.sample_rate:
    #         waveform = T.Resample(sr, self.sample_rate)(waveform)
    #     # make mono
    #     if waveform.shape[0] > 1:
    #         waveform = waveform.mean(dim=0, keepdim=True)
    #     # pad/truncate
    #     if waveform.shape[1] > self.max_samples:
    #         waveform = waveform[:, :self.max_samples]
    #     else:
    #         pad = self.max_samples - waveform.shape[1]
    #         waveform = F.pad(waveform, (0, pad))
    #     return waveform.squeeze(0)

    def load_audio(self, path):
        try:
            # First try default backend (could be sox_io or torchcodec)
            waveform, sr = torchaudio.load(path)
        except Exception:
            # If torchcodec / backend fails, fallback to soundfile
            import soundfile as sf
            import numpy as np

            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            if audio.ndim == 1:
                audio = audio[None, :]  # (1, N)
            else:
                audio = audio.T  # (channels, samples)
            waveform = torch.from_numpy(audio)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = T.Resample(sr, self.sample_rate)(waveform)
            sr = self.sample_rate

        # Make mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or truncate
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        else:
            pad = self.max_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad))

        return waveform.squeeze(0)


    def apply_augment(self, waveform):
        if not self.augment:
            return waveform
        # RawBoost-style: small gaussian noise
        if random.random() < 0.5:
            waveform = waveform + torch.randn_like(waveform) * 0.003
        return waveform

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        label = self.labels[idx]
        waveform = self.load_audio(path)
        waveform = self.apply_augment(waveform)

        # Structural: mel-spectrogram (shape: n_mels x time)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-7)

        # Artifact branch: high-fidelity spectrogram (freq_bins x time)
        spec = self.spec_transform(waveform)
        spec = torch.log(spec + 1e-7)

        # Apply spectrogram augmentations if requested
        if self.augment:
            if random.random() < 0.5:
                mel = self.freq_mask(mel)
            if random.random() < 0.5:
                spec = self.freq_mask(spec)
        return {"mel": mel, "spec": spec, "label": torch.tensor(label, dtype=torch.long),
                "path": path}
    


# -------------------------
# Utilities: ASV Spoof dataset exploration and preparation
# -------------------------
def explore_asvspoof_structure(base_path):
    print("ASV Spoof Dataset Structure Exploration:")
    print("=" * 50)
    if not os.path.exists(base_path):
        print("Base path does not exist:", base_path)
        return
    print("Base path:", base_path)
    
    # Check for main directory structure
    la_path = os.path.join(base_path, "LA", "LA")
    if os.path.exists(la_path):
        print("Found LA directory")
        contents = os.listdir(la_path)
        print("LA contents:", contents)
        
        # Check for protocol files
        protocol_dir = os.path.join(la_path, "ASVspoof2019_LA_cm_protocols")
        if os.path.exists(protocol_dir):
            protocols = os.listdir(protocol_dir)
            print("Protocol files:", protocols)
        
        # Check for audio directories
        for split in ["ASVspoof2019_LA_train", "ASVspoof2019_LA_dev", "ASVspoof2019_LA_eval"]:
            split_path = os.path.join(la_path, split)
            if os.path.exists(split_path):
                print(f"Found {split}")
                flac_path = os.path.join(split_path, "flac")
                if os.path.exists(flac_path):
                    files = [f for f in os.listdir(flac_path) if f.endswith(".flac")]
                    print(f"  {split}/flac: {len(files)} files")
                    if files:
                        print(f"    Sample files: {files[:3]}")
    print("=" * 50)

def prepare_asvspoof_data(data_path, split="train"):
    """
    Prepare ASV Spoof 2019 data
    split: "train", "dev", or "eval"
    """
    la_path = os.path.join(data_path, "LA", "LA")
    
    # Protocol file mapping
    protocol_files = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt", 
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt"
    }
    
    # Audio directory mapping
    audio_dirs = {
        "train": "ASVspoof2019_LA_train",
        "dev": "ASVspoof2019_LA_dev",
        "eval": "ASVspoof2019_LA_eval"
    }
    
    protocol_path = os.path.join(la_path, "ASVspoof2019_LA_cm_protocols", protocol_files[split])
    audio_dir = os.path.join(la_path, audio_dirs[split], "flac")
    
    if not os.path.exists(protocol_path):
        print(f"Protocol file not found: {protocol_path}")
        return [], []
    
    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        return [], []
    
    audio_paths = []
    labels = []
    
    # Read protocol file
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                audio_file = parts[1]
                key = parts[4] if len(parts) > 4 else parts[-1]
                
                # Convert to binary label (0=bonafide/real, 1=spoof/fake)
                label = 0 if key.lower() == 'bonafide' else 1
                
                # Build full path
                audio_path = os.path.join(audio_dir, f"{audio_file}.flac")
                if os.path.exists(audio_path):
                    audio_paths.append(audio_path)
                    labels.append(label)
    
    print(f"Loaded {split} split: {len(audio_paths)} files")
    if labels:
        print(f"  Bonafide: {labels.count(0)}, Spoof: {labels.count(1)}")
    
    return audio_paths, labels
