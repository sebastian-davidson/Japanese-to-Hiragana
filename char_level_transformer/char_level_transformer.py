import torch
import torch.nn as nn
import os
import math # for math.log

from common.common import encode_sentence, calculate_cer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

        
class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, device, d_model=256, nhead=4, num_layers=3, dropout=0.1, max_len=64):
        super().__init__()
        self.device = device
        self.src_tok_emb = nn.Embedding(input_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(output_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            dim_feedforward=512,
            batch_first=True
        )

        self.generator = nn.Linear(d_model, output_vocab_size)

    def forward(self, src, tgt, src_pad_mask=None, tgt_mask=None, tgt_pad_mask=None):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_pad_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        return self.generator(output)

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))


BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

def train_model(model, dataloader, scheduler, optimizer, criterion, pad_token, dataset
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

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(model.device)
            src_pad_mask = (src == pad_token)
            tgt_pad_mask = (tgt_input == pad_token)

            output = model(src, tgt_input, src_pad_mask=src_pad_mask, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
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
            'input2idx': dataset.input2idx,
            'output2idx': dataset.output2idx,
            'idx2output': dataset.idx2output,
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


"""
# Inference function (using greedy search)
def predict(model, sentence, input2idx, output2idx, idx2output, device, max_len=64):
    model.eval()
    sos_idx = output2idx['<sos>']
    eos_idx = output2idx['<eos>']
    pad_idx = output2idx['<pad>']

    src = torch.tensor([encode_sentence(sentence, input2idx, max_len)], dtype=torch.long).to(device)
    src_mask = (src == input2idx['<pad>'])

    output = torch.tensor([[sos_idx]], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            tgt_mask = model.generate_square_subsequent_mask(output.size(1)).to(device)
            out = model(src, output, src_pad_mask=src_mask, tgt_mask=tgt_mask)
            next_token = out[:, -1, :].argmax(-1).item()
            output = torch.cat([output, torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == eos_idx:
                break

    return ''.join(idx2output[idx.item()] for idx in output[0][1:] if idx.item() != eos_idx and idx.item() != pad_idx)
"""


# Inference function (using beam search)
def predict(model, sentence, input2idx, output2idx, idx2output, device, max_len=64, beam_width=3):
    model.eval()
    sos_idx = output2idx['<sos>']
    eos_idx = output2idx['<eos>']
    pad_idx = output2idx['<pad>']

    with torch.no_grad():
        src = torch.tensor([encode_sentence(sentence, input2idx, max_len)], dtype=torch.long).to(device)
        src_mask = (src == input2idx['<pad>'])

        # Initialize beam: (sequence, log prob)
        beams = [([sos_idx], 0.0)]

        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                tgt = torch.tensor([seq], dtype=torch.long).to(device)
                tgt_mask = model.generate_square_subsequent_mask(len(seq)).to(device)
                out = model(src, tgt, src_pad_mask=src_mask, tgt_mask=tgt_mask)
                log_probs = torch.log_softmax(out[0, -1], dim=-1)  # (vocab_size,)

                topk = torch.topk(log_probs, beam_width)
                for i in range(beam_width):
                    token = topk.indices[i].item()
                    new_seq = seq + [token]
                    new_score = score + topk.values[i].item()
                    new_beams.append((new_seq, new_score))

            # Keep top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Early stop if all beams end with <eos>
            if all(seq[-1] == eos_idx for seq, _ in beams):
                break

        # Pick the best beam
        best_seq = beams[0][0]

        return ''.join(
            idx2output[idx] for idx in best_seq[1:] if idx not in (eos_idx, pad_idx)
        )
