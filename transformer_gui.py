import sys
import os

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QTextEdit, QPushButton, QLabel, QSizePolicy
)
from PyQt6.QtCore import Qt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.common import KanjiHiraganaDataset, read_tsv_to_tuples, build_vocab
from char_level_transformer.char_level_transformer import TransformerSeq2Seq, train_model, predict

def translate_to_hiragana(japanese_text):
    global model, input2idx, output2idx, idx2output, DEVICE, MAX_LEN
    if not japanese_text:
        return ""
    return predict(model, japanese_text, input2idx, output2idx, idx2output, DEVICE, MAX_LEN)


class TranslatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Japanese Kanji to Hiragana Translator')

        # Main vertical layout
        layout = QVBoxLayout()

        # Input Text Area
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("ここに日本語の文章を入力してください...") # Enter Japanese sentences here...
        self.input_text.setMinimumHeight(100)
        layout.addWidget(QLabel("Input Japanese Text:")) # Label for input box
        layout.addWidget(self.input_text)

        # Translate Button
        self.translate_button = QPushButton('ひらがなに変換 (Translate to Hiragana)')
        self.translate_button.clicked.connect(self.on_translate_click)
        layout.addWidget(self.translate_button)

        self.output_label = QLabel("ここに変換結果が表示されます...") # Translation result appears here...
        self.output_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px;")
        self.output_label.setWordWrap(True) # Allow text wrapping
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignTop) # Align text to top

        self.output_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.output_label.setMinimumHeight(100)

        layout.addWidget(QLabel("Output (Hiragana):")) # Label for output box
        layout.addWidget(self.output_label) # or self.output_text

        # Set the layout for the main window
        self.setLayout(layout)

        # Set initial window size
        self.resize(500, 400)

        # Show the window
        self.show()

    def on_translate_click(self):
        """
        Called when the translate button is clicked.
        """
        # Get text from the input QTextEdit
        japanese_text = self.input_text.toPlainText().strip() # Get text and remove leading/trailing whitespace

        if not japanese_text:
            # If input is empty, show a message in the output area
            self.output_label.setText("テキストを入力してください。") # Please enter text.
            return

        try:
            hiragana_result = translate_to_hiragana(japanese_text)

            self.output_label.setText(hiragana_result)

        except Exception as e:
            error_message = f"翻訳中にエラーが発生しました。\nError: {str(e)}"
            print(f"Error during translation call: {e}")
            self.output_label.setText(error_message)


if __name__ == '__main__':
    checkpoint_path = './char_level_transformer/checkpoints/best_model.pt'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 64
    EMB_SIZE = 256

    if not os.path.exists(checkpoint_path):
        print('Model not available', file=sys.stderr)
        sys.exit(1)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    input2idx = checkpoint['input2idx']
    output2idx = checkpoint['output2idx']
    idx2output = checkpoint['idx2output']

    model = TransformerSeq2Seq(
        input_vocab_size=len(input2idx),
        output_vocab_size=len(output2idx),
        device=DEVICE,
        d_model=EMB_SIZE,
        max_len=MAX_LEN,
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Now actually create the Qt Application and the main window.
    app = QApplication(sys.argv)
    ex = TranslatorApp()

    # Then execute the application.
    sys.exit(app.exec())
