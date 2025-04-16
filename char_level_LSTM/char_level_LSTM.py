import torch
import torch.nn as nn
import random
import os

from common.common import encode_sentence, calculate_cer

# Bidirectional encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.embedding(x)
        outputs, (h, c) = self.lstm(x)
        # h and c are (2, batch, hidden_size) -> concatenate forward and backward
        h = torch.cat((h[0], h[1]), dim=1).unsqueeze(0) # (1, batch, hidden_size*2)
        c = torch.cat((c[0], c[1]), dim=1).unsqueeze(0)
        return h, c


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        # Match encoder output size, which is hidden_size*2
        self.lstm = nn.LSTM(emb_size, hidden_size * 2, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, h, c):
        x = self.embedding(x).unsqueeze(1)  # (batch, 1, emb)
        out, (h, c) = self.lstm(x, (h, c))  # h, c = (1, batch, hidden*2)
        out = self.fc(out.squeeze(1))       # (batch, vocab_size)
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
        input = trg[:, 0] # will be <sos> token

        for t in range(1, trg_len):
            output, h, c = self.decoder(input, h, c)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if random.random() < teacher_forcing_ratio else top1
        return outputs


BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

def train_model(model, dataloader, scheduler, optimizer, criterion, pad_token,
                num_epochs=100, patience_limit=5, save_every=5, start_epoch=0):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_loss = float('inf')
    patience_counter = 0
    loss_values = []

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
            torch.save(checkpoint_data, os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pt"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(checkpoint_data, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print("New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break
        scheduler.step(avg_loss)


# Inference function with beam search, which is standard in practice
def predict(model, sentence, input2idx, output2idx, idx2output, max_len, beam_width=3):
    model.eval()
    sos_token = output2idx['<sos>']
    eos_token = output2idx['<eos>']
    pad_token = output2idx['<pad>']
    device = model.device

    with torch.no_grad():
        # Encode the input sentence
        src_tensor = torch.tensor([encode_sentence(sentence, input2idx, max_len)], dtype=torch.long).to(device)
        h, c = model.encoder(src_tensor)

        # Beam candidates: (sequence, log probability, hidden, cell)
        beams = [([sos_token], 0.0, h, c)]

        for _ in range(max_len):
            new_beams = []
            for seq, log_prob, h, c in beams:
                input_token = torch.tensor([seq[-1]], device=device)
                output, h_new, c_new = model.decoder(input_token, h, c)
                probs = torch.log_softmax(output, dim=1).squeeze(0)  # (vocab_size,)

                topk_probs, topk_idxs = torch.topk(probs, beam_width)

                for i in range(beam_width):
                    next_token = topk_idxs[i].item()
                    total_log_prob = log_prob + topk_probs[i].item()
                    new_seq = seq + [next_token]
                    new_beams.append((new_seq, total_log_prob, h_new, c_new))

            # Keep top k sequences
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Check for end-of-sequence in all beams
            if all(seq[-1] == eos_token for seq, *_ in beams):
                break

        # Choose the best completed sequence (or best overall)
        final_seq = beams[0][0]

        # Convert to characters
        result = ''.join(idx2output[idx] for idx in final_seq[1:] if idx not in (eos_token, pad_token))
        return result
