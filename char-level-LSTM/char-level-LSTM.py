import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import csv
import os
import argparse

# Load Dataset
def read_tsv_to_tuples(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                row = tuple(line.strip().split('\t'))
                data.append(row)
    except Exception as e:
        print(e)
        return []
    return data

# Vocabulary setup
def build_vocab(dataset_pairs):
    input_strings, output_strings = zip(*dataset_pairs)
    input_chars = {c for s in input_strings for c in s}
    output_chars = {c for s in output_strings for c in s}

    special_tokens = ['<pad>', '<sos>', '<eos>']
    input_vocab = special_tokens + list(input_chars)
    output_vocab = special_tokens + list(output_chars)

    input2idx = {c: i for i, c in enumerate(input_vocab)}
    output2idx = {c: i for i, c in enumerate(output_vocab)}
    idx2output = {i: c for c, i in output2idx.items()}
    return input2idx, output2idx, idx2output, output2idx['<pad>']

# Encoding function
def encode_sentence(sentence, vocab, max_len):
    tokens = [vocab['<sos>']] + [vocab[c] for c in sentence] + [vocab['<eos>']]
    tokens += [vocab['<pad>']] * (max_len - len(tokens))
    return tokens[:max_len]

# Custom Dataset
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

class LSTMSeq2Seq(nn.Module):
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

def train_model(model, dataloader, scheduler, optimizer, criterion, pad_token,
                num_epochs=100, patience_limit=5, save_every=5, start_epoch=0):
    best_loss = float('inf')
    patience_counter = 0
    loss_values = []
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0

        for src, tgt in dataloader:
            src, tgt = src.to(model.device), tgt.to(model.device)
            optimizer.zero_grad()
            output = model(src, tgt)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_values.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'input2idx': input2idx,
            'output2idx': output2idx,
            'idx2output': idx2output,
            'epoch': epoch + 1,
            'loss': avg_loss
        }

        if (epoch + 1) % save_every == 0:
            torch.save(checkpoint_data, f"checkpoints/model_epoch_{epoch+1}.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(checkpoint_data, "checkpoints/best_model.pt")
            print("New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break
        scheduler.step(avg_loss)

# Inference function
def predict(model, sentence, input2idx, output2idx, idx2output, max_len=64):
    model.eval()
    with torch.no_grad():
        tokens = encode_sentence(sentence, input2idx, max_len)
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(model.device)
        h, c = model.encoder(src_tensor)
        input_token = torch.tensor([output2idx['<sos>']]).to(model.device)

        result = []
        for _ in range(max_len):
            output, h, c = model.decoder(input_token, h, c)
            top1 = output.argmax(1).item()
            char = idx2output[top1]
            if char == '<eos>':
                break
            result.append(char)
            input_token = torch.tensor([top1]).to(model.device)
        return ''.join(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kanji to Hiragana Converter")
    parser.add_argument('-t', '--train', action='store_true', help="Train the model (default is to skip training)")
    parser.add_argument('-p', '--path', type=str, default='checkpoints/best_model.pt',
                        help='The file path to the model to use')
    args = parser.parse_args()

    dataset_pairs = read_tsv_to_tuples('../kanji_hiragana_pairs.tsv')
    checkpoint_path = args.path
    start_epoch = 0

    # Try to load vocab + model if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"Loading saved model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        input2idx = checkpoint['input2idx']
        output2idx = checkpoint['output2idx']
        idx2output = checkpoint['idx2output']
        PAD_token = output2idx['<pad>']
    else:
        print("No saved model found, building vocab from dataset.")
        input2idx, output2idx, idx2output, PAD_token = build_vocab(dataset_pairs)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMB_SIZE = 256
    HID_SIZE = 512
    MAX_LEN = 64
    BATCH_SIZE = 128
    NUM_EPOCHS = 500
    PATIENCE_LIMIT = 5
    SAVE_EVERY = 5

    encoder = Encoder(len(input2idx), EMB_SIZE, HID_SIZE)
    decoder = Decoder(len(output2idx), EMB_SIZE, HID_SIZE)
    model = LSTMSeq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)

    train_dataset = KanjiHiraganaDataset(dataset_pairs, input2idx, output2idx, max_len=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    if args.train or not os.path.exists(checkpoint_path):
        train_model(model, train_loader, scheduler, optimizer, criterion, PAD_token,
                    num_epochs=NUM_EPOCHS, patience_limit=PATIENCE_LIMIT,
                    save_every=SAVE_EVERY, start_epoch=start_epoch)
    else:
        print("Skipping training (use --train to force training)")

    test_sentences = [
        ("明日は雨です","あすはあめです。"),
        ("彼は学生です","かれはがくせいです。"),
        ("彼は生きている間に生け花を生業とした", "かれはいきているまにいけばなをなりわいとした。")
    ]
    for sentence, expected_output in test_sentences:
        print(f"Input: {sentence} => Output: {predict(model, sentence, input2idx, output2idx, idx2output, MAX_LEN)}")
        print(f"Expected output: {expected_output}")
