import torch
from torch.utils.data import Dataset, DataLoader
import Levenshtein


# Load Dataset; it's assumed this is a 2-column .tsv file of sentence pairs
def read_tsv_to_tuples(file_path):
    """
    Arguments:
        file_path: the absolute path to the 2-column .tsv file
    Returns:
        a list of tuples representing each row
    """
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
    """
    Arguments:
        dataset_pairs:
            a list of tuples representing the input and desired output
    Returns:
        a dictionary mapping input chars to the number assigned to them,
        a dictionary mapping outnput chars to the number assigned to them,
        a dictionary that is the inverse of the former dictionary
    """
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
    def __init__(self, pairs, input2idx, output2idx, idx2output, max_len=20):
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



def calculate_cer(predictions, references):
    """
    Compute average Character Error Rate (CER) over a list of predicted and reference strings.
    Arguments:
        predictions: what the model actually outputs
        references: the desired output
    Returns:
        the average CER across all samples
    """
    assert len(predictions) == len(references), "Mismatched prediction/reference count"
    total_errors = 0
    total_chars = 0

    for pred, ref in zip(predictions, references):
        total_errors += Levenshtein.distance(pred, ref)
        total_chars += len(ref)

    return total_errors / total_chars if total_chars != 0 else float('inf')

