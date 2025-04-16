import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.common import KanjiHiraganaDataset, read_tsv_to_tuples, build_vocab
from char_level_transformer.char_level_transformer import TransformerSeq2Seq, train_model, predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kanji to Hiragana Converter")
    parser.add_argument('-t', '--train', action='store_true', help="Train the model (default is to skip training)")
    parser.add_argument('-p', '--path', type=str, default='./char_level_transformer/checkpoints/best_model.pt',
                        help='The file path to the model to use')
    args = parser.parse_args()

    dataset_pairs = read_tsv_to_tuples('kanji_hiragana_pairs.tsv')
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
    BATCH_SIZE = 512
    NUM_EPOCHS = 500
    PATIENCE_LIMIT = 5
    SAVE_EVERY = 5

    model = TransformerSeq2Seq(
        input_vocab_size=len(input2idx),
        output_vocab_size=len(output2idx),
        device=DEVICE,
        d_model=EMB_SIZE,
        max_len=MAX_LEN,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
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
        print(f"Input: {sentence} => Output: {predict(model, sentence, input2idx, output2idx, idx2output, DEVICE, MAX_LEN)}")
        print(f"Expected output: {expected_output}")
