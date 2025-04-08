# Kanji to Hiragana Converter: Character-level Seq2Seq Model (PyTorch)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import csv

"""
# ----------------------
# Sample toy dataset
# ----------------------
dataset_pairs = [
    ("明日は雨です", "あしたはあめです"),
    ("東京に行きます", "とうきょうにいきます"),
    ("彼は学生です", "かれはがくせいです"),
    ("私は日本人です", "わたしはにほんじんです")
]
"""

def read_tsv_to_tuples(file_path):
    """
    Reads a TSV file and returns a list of tuples, where each tuple 
    represents a row in the file.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                row = tuple(line.strip().split('\t'))
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return []
    except Exception as e:
         print(f"An error occurred: {e}")
         return []
    return data

dataset_pairs = read_tsv_to_tuples('./kanji_hiragana_pairs.tsv')

# ----------------------
# Vocabulary setup
# ----------------------
input_chars = set()
output_chars = set()

for src, tgt in dataset_pairs:
    input_chars.update(src)
    output_chars.update(tgt)

input_vocab = ['<pad>', '<sos>', '<eos>'] + sorted(input_chars)
output_vocab = ['<pad>', '<sos>', '<eos>'] + sorted(output_chars)

input2idx = {c: i for i, c in enumerate(input_vocab)}
output2idx = {c: i for i, c in enumerate(output_vocab)}
idx2output = {i: c for c, i in output2idx.items()}

# ----------------------
# Encoding function
# ----------------------
def encode_sentence(sentence, vocab, max_len):
    tokens = [vocab['<sos>']] + [vocab[c] for c in sentence] + [vocab['<eos>']]
    tokens += [vocab['<pad>']] * (max_len - len(tokens))
    return tokens[:max_len]

# ----------------------
# Custom Dataset
# ----------------------
class KanjiHiraganaDataset(Dataset):
    def __init__(self, pairs, input2idx, output2idx, max_len=20):
        self.data = pairs
        self.input2idx = input2idx
        self.output2idx = output2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_encoded = torch.tensor(encode_sentence(src, self.input2idx, self.max_len))
        tgt_encoded = torch.tensor(encode_sentence(tgt, self.output2idx, self.max_len))
        return src_encoded, tgt_encoded

# ----------------------
# Seq2Seq Model
# ----------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        outputs, (h, c) = self.lstm(x)
        return h, c

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h, c):
        x = self.embedding(x).unsqueeze(1)
        out, (h, c) = self.lstm(x, (h, c))
        out = self.fc(out.squeeze(1))
        return out, h, c

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        h, c = self.encoder(src)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, h, c = self.decoder(input, h, c)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if random.random() < teacher_forcing_ratio else top1
        return outputs

# ----------------------
# Training setup
# ----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_SIZE = 64
HID_SIZE = 128
MAX_LEN = 20

train_dataset = KanjiHiraganaDataset(dataset_pairs, input2idx, output2idx, max_len=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

encoder = Encoder(len(input_vocab), EMB_SIZE, HID_SIZE)
decoder = Decoder(len(output_vocab), EMB_SIZE, HID_SIZE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=output2idx['<pad>'])

# Training loop
for epoch in range(30):
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, tgt)
        output = output[:, 1:].reshape(-1, output.shape[-1])
        tgt = tgt[:, 1:].reshape(-1)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Inference function
def predict(model, sentence):
    model.eval()
    with torch.no_grad():
        tokens = encode_sentence(sentence, input2idx, MAX_LEN)
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)
        h, c = model.encoder(src_tensor)
        input_token = torch.tensor([output2idx['<sos>']]).to(DEVICE)

        result = []
        for _ in range(MAX_LEN):
            output, h, c = model.decoder(input_token, h, c)
            top1 = output.argmax(1).item()
            char = idx2output[top1]
            if char == '<eos>':
                break
            result.append(char)
            input_token = torch.tensor([top1]).to(DEVICE)
        return ''.join(result)

# Test the model
test_sentences = ["明日は雨です", "彼は学生です"]
for sentence in test_sentences:
    print(f"Input: {sentence} => Output: {predict(model, sentence)}")

