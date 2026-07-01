# MULTIMODAL EMOTION RECOGNITION SYSTEM
# Audio + Text 
# Dataset : IEMOCAP
# IMPORT REQUIRED LIBRARIES
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

# SETTINGS
DATASET_PATH = "D:/EmotionProject/IEMOCAP_full_release"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nStage 1: Starting Multimodal Emotion Recognition")
print("Using Device:", device)

# Emotion label mapping
label_map = {"ang":0,"hap":1,"sad":2,"neu":3,"exc":1}
emotion_names = ["Angry","Happy","Sad","Neutral"]

# DATASET SCANNER
# Scans all folders to collect
# 1) Audio files
# 2) Transcripts
# 3) Emotion labels
def scan_dataset(root):
    wav_files = {}
    transcripts = {}
    labels = {}
    session_set = set()
    print("\nStage 2: Scanning dataset recursively...\n")
    for root_dir, dirs, files in os.walk(root):
        for s in ["Session1","Session2","Session3","Session4","Session5"]:
            if s in root_dir:
                session_set.add(s)
        for file in files:
            # Ignore Mac metadata files
            if file.startswith("._"):
                continue
            path = os.path.join(root_dir,file)
            # WAV FILES
            if file.lower().endswith(".wav"):
                utt_id = file.replace(".wav","").strip()
                wav_files[utt_id] = path
            # TRANSCRIPTS
            if "transcription" in root_dir.lower():
                try:
                    with open(path,"r",encoding="latin-1") as f:
                        for line in f:
                            line=line.strip()
                            if len(line)==0:
                                continue
                            parts=line.split()
                            utt_id=parts[0]
                            if ":" in line:
                                text=line.split(":",1)[1].strip()
                            else:
                                text=""

                            transcripts[utt_id]=text
                except:
                    pass

            # LABEL FILES 
            if "emoevaluation" in root_dir.lower():
                try:
                    with open(path,"r",encoding="latin-1") as f:

                        for line in f:

                            if line.startswith("["):

                                parts=line.split("\t")

                                if len(parts)>=3:

                                    utt_id=parts[1].strip()
                                    emo=parts[2].strip().lower()

                                    if emo in label_map:
                                        labels[utt_id]=emo

                except:
                    pass
    print("Sessions detected:",len(session_set))
    print("Total WAV files found:",len(wav_files))
    print("Total transcripts found:",len(transcripts))
    print("Total emotion labels found:",len(labels))
    return wav_files, transcripts, labels

# Convert speech signal into machine learning features
def extract_audio_features(audio_path):
    # Load audio
    y,sr = librosa.load(audio_path,sr=16000)
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40)
    # Extract chroma features
    chroma = librosa.feature.chroma_stft(y=y,sr=sr)
    # Extract mel spectrogram
    mel = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128)
    # Combine all features
    features = np.vstack([mfcc,chroma,mel])
    # Padding / truncation for fixed size
    max_len = 200
    if features.shape[1] < max_len:
        pad = max_len - features.shape[1]
        features = np.pad(features,((0,0),(0,pad)))
    else:
        features = features[:,:max_len]
    return features.T

# DATASET CLASS
# Combines Audio + Text + Emotion Labels
class EmotionDataset(Dataset):
    def __init__(self):
        wav_files, transcripts, labels = scan_dataset(DATASET_PATH)
        self.samples=[]
        print("\nStage 3: Matching audio + text + labels\n")
        for utt_id in wav_files:
            if utt_id in transcripts and utt_id in labels:
                wav_path = wav_files[utt_id]
                text = transcripts[utt_id]
                label = labels[utt_id]
                self.samples.append((wav_path,text,label))
        print("Total matched samples:",len(self.samples))
        # Text preprocessing using BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        wav_path,text,emo = self.samples[idx]
        # Audio preprocessing
        audio = extract_audio_features(wav_path)
        audio = torch.tensor(audio).float()
        # Text preprocessing
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=50,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        mask = encoded["attention_mask"].squeeze(0)
        label = torch.tensor(label_map[emo])
        return audio,input_ids,mask,label

# MULTIMODAL MODEL
# Audio LSTM + Text BERT + Fusion Layer
class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Text encoder
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # Audio encoder
        self.audio_lstm = nn.LSTM(180,128,batch_first=True,bidirectional=True)
        # Text sequence model
        self.text_lstm = nn.LSTM(768,128,batch_first=True,bidirectional=True)
        # Fusion layer
        self.fc = nn.Linear(512,4)
    def forward(self,audio,input_ids,mask):
        audio_out,_ = self.audio_lstm(audio)
        audio_feat = torch.mean(audio_out,dim=1)
        bert_out = self.bert(input_ids=input_ids,attention_mask=mask)
        text_embed = bert_out.last_hidden_state
        text_out,_ = self.text_lstm(text_embed)
        text_feat = torch.mean(text_out,dim=1)
        combined = torch.cat((audio_feat,text_feat),dim=1)
        return self.fc(combined)

# LOAD DATASET
dataset = EmotionDataset()
print("\nStage 4: Dataset Loaded Successfully")
print("Total Samples:",len(dataset))
train_size=int(0.8*len(dataset))
test_size=len(dataset)-train_size
print("Train Samples:",train_size)
print("Test Samples:",test_size)
train_dataset,test_dataset=random_split(dataset,[train_size,test_size])
train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=16)

# MODEL OPTIMIZATION SETTINGS
labels=[label_map[s[2]] for s in dataset.samples]
counts=Counter(labels)

# Handle class imbalance
weights=torch.tensor([1.0/counts[i] for i in range(4)]).to(device)
model=MultimodalModel().to(device)
criterion=nn.CrossEntropyLoss(weight=weights)
optimizer=optim.AdamW(model.parameters(),lr=2e-5)
EPOCHS=50

# TRAINING LOOP
print("\nStage 5: Training Started\n")
for epoch in range(EPOCHS):
    model.train()
    preds_all=[]
    labels_all=[]
    for audio,input_ids,mask,label in train_loader:
        audio=audio.to(device)
        input_ids=input_ids.to(device)
        mask=mask.to(device)
        label=label.to(device)
        optimizer.zero_grad()
        outputs=model(audio,input_ids,mask)
        loss=criterion(outputs,label)
        # Backpropagation
        loss.backward()
        # Weight update
        optimizer.step()
        preds=torch.argmax(outputs,dim=1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(label.cpu().numpy())
    acc=accuracy_score(labels_all,preds_all)
    print(f"Epoch {epoch+1}/{EPOCHS} Accuracy: {acc*100:.2f}%")

# MODEL EVALUATION
print("\nStage 6: Evaluation\n")
model.eval()
all_preds=[]
all_labels=[]
with torch.no_grad():
    for audio,input_ids,mask,label in test_loader:
        audio=audio.to(device)
        input_ids=input_ids.to(device)
        mask=mask.to(device)
        outputs=model(audio,input_ids,mask)
        preds=torch.argmax(outputs,dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.numpy())

accuracy=accuracy_score(all_labels,all_preds)
print("Final Accuracy:",accuracy*100)
print("\nClassification Report:\n")
print(classification_report(all_labels,all_preds,target_names=emotion_names))

# CONFUSION MATRIX
print("\nStage 7: Confusion Matrix")
cm=confusion_matrix(all_labels,all_preds)
plt.figure(figsize=(6,5))
plt.imshow(cm,interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
classes=["Angry","Happy","Sad","Neutral"]
tick_marks=np.arange(len(classes))
plt.xticks(tick_marks,classes)
plt.yticks(tick_marks,classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,cm[i,j],ha="center",va="center")
plt.tight_layout()
plt.show()


# USER EMOTION DETECTION
def record_audio(filename="user_voice.wav",duration=5,sr=16000):
    print("\nSpeak now...")
    recording=sd.rec(int(duration*sr),samplerate=sr,channels=1)
    sd.wait()
    recording=recording.squeeze()
    sf.write(filename,recording,sr)
    print("Recording saved")
    return filename

def predict_emotion(audio_path,text):
    audio=extract_audio_features(audio_path)
    audio=torch.tensor(audio).float().unsqueeze(0).to(device)
    encoded=dataset.tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=50,
        return_tensors="pt"
    )
    input_ids=encoded["input_ids"].to(device)
    mask=encoded["attention_mask"].to(device)
    model.eval()
    with torch.no_grad():
        outputs=model(audio,input_ids,mask)
        pred=torch.argmax(outputs,dim=1).item()
    emotions=["Angry","Happy","Sad","Neutral"]
    return emotions[pred]
print("\nStage 8: User Emotion Detection")

while True:
    text=input("\nEnter what you said (type 'exit' to stop): ")
    if text.lower()=="exit":
        break
    audio_file=record_audio()
    emotion=predict_emotion(audio_file,text)
    print("\nPredicted Emotion:",emotion)