# 1. IMPORTS                                  
import os, re, math
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
from sentence_transformers import SentenceTransformer
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import seaborn as sns
                                        
# 2. PATHS                                       
base_path = "/content"
video_root = os.path.join(base_path, "videos")
wav_root = os.path.join(base_path, "wav_files")
data_path = os.path.join(base_path, "data")
csv_file = os.path.join(data_path, "dataset.csv")
os.makedirs(wav_root, exist_ok=True)
                                          
# 3. LOAD DATA                                         
df = pd.read_csv(csv_file)
df = df[['Utterance','Emotion','Dialogue_ID','Utterance_ID']].dropna()
                                          
# 4. MP4 TO WAV                                      
audio_map = {}
for root, _, files in os.walk(video_root):
    for file in files:
        if file.endswith(".mp4"):
            match = re.search(r'dia(\d+)_utt(\d+)', file.lower())
            if match:
                key = f"{int(match.group(1))}_{int(match.group(2))}"
                video_path = os.path.join(root, file)
                wav_path = os.path.join(wav_root, key + ".wav")
                if not os.path.exists(wav_path):
                    video = VideoFileClip(video_path)
                    video.audio.write_audiofile(wav_path, verbose=False, logger=None)
                audio_map[key] = wav_path

                                          
# 5. TEXT FEATURES                                          
bert = SentenceTransformer('paraphrase-mpnet-base-v2')
X_text = bert.encode(df['Utterance'].tolist(), show_progress_bar=True)

                                          
# 6. AUDIO FEATURES                                          
def extract_logmel_patches(file, sr=16000, n_mels=64, patch_frames=16, max_patches=32):
    y, sr = librosa.load(file, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel)
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-6)
    patches = []
    for i in range(0, logmel.shape[1] - patch_frames, patch_frames):
        patch = logmel[:, i:i+patch_frames].flatten()
        patches.append(patch)
    patches = patches[:max_patches]
    while len(patches) < max_patches:
        patches.append(np.zeros(n_mels * patch_frames))
    return np.array(patches)
X_audio = []
valid_idx = []
for i, row in df.iterrows():
    key = f"{row['Dialogue_ID']}_{row['Utterance_ID']}"
    if key in audio_map:
        X_audio.append(extract_logmel_patches(audio_map[key]))
        valid_idx.append(i)
df = df.iloc[valid_idx]
X_text = X_text[valid_idx]
X_audio = np.array(X_audio)
                                          
# 7. LABEL ENCODING                                  
le = LabelEncoder()
y = le.fit_transform(df['Emotion'])
                                    
# 8. DATA SPLIT                                         
X_text_train, X_text_test, X_audio_train, X_audio_test, y_train, y_test = train_test_split(
    X_text, X_audio, y, test_size=0.2, random_state=42)
X_text_train, X_text_val, X_audio_train, X_audio_val, y_train, y_val = train_test_split(
    X_text_train, X_audio_train, y_train, test_size=0.1, random_state=42)

                                          
# 9. CLASS IMBALANCE HANDLING                                    
class_counts = Counter(y_train)
total = sum(class_counts.values())
weights = []
for i in range(len(le.classes_)):
    weights.append(total / (len(le.classes_) * class_counts[i]))
weights = torch.tensor(weights, dtype=torch.float32)
                                    
# 10. DATASET                               
class MERDataset(Dataset):
    def __init__(self, X_text, X_audio, y):
        self.X_text = torch.tensor(X_text, dtype=torch.float32)
        self.X_audio = torch.tensor(X_audio, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X_text[idx], self.X_audio[idx], self.y[idx]
train_loader = DataLoader(MERDataset(X_text_train, X_audio_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(MERDataset(X_text_val, X_audio_val, y_val), batch_size=32)
test_loader = DataLoader(MERDataset(X_text_test, X_audio_test, y_test), batch_size=32)
                                          
# 11. AUDIO TRANSFORMER                                          
class AudioTransformer(nn.Module):
    def __init__(self, input_dim=1024, d_model=256, nhead=4, num_layers=2):
        super().__init__()
        self.patch_embed = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        B, N, D = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.encoder(x)
        return x[:, 0]

                                          
# 12. MODEL
class MultimodalModel(nn.Module):
    def _init_(self, text_dim=768, num_classes=6):
        super()._init_()
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        self.audio_model = AudioTransformer()
        self.gate = nn.Linear(512, 1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, text, audio):
        t = self.text_fc(text)
        a = self.audio_model(audio)
        g = torch.sigmoid(self.gate(torch.cat([t, a], dim=1)))
        fused = g * a + (1 - g) * t
        out = self.classifier(fused)
        return out

                                          
# 13. TRAINING                                         
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalModel(num_classes=len(le.classes_)).to(device)
weights = weights.to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
best_val_acc = 0
for epoch in range(20):
    model.train()
    for text, audio, labels in train_loader:
        text, audio, labels = text.to(device), audio.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(text, audio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for text, audio, labels in val_loader:
            text, audio = text.to(device), audio.to(device)
            outputs = model(text, audio)
            preds.extend(outputs.argmax(1).cpu().numpy())
            trues.extend(labels.numpy())
    val_acc = accuracy_score(trues, preds)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

                                          
# 14. TESTING                                          
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
preds, trues = [], []
with torch.no_grad():
    for text, audio, labels in test_loader:
        text, audio = text.to(device), audio.to(device)
        outputs = model(text, audio)
        preds.extend(outputs.argmax(1).cpu().numpy())
        trues.extend(labels.numpy())
print("Test Accuracy:", accuracy_score(trues, preds))
cm = confusion_matrix(trues, preds)
sns.heatmap(cm, annot=True)
plt.show()