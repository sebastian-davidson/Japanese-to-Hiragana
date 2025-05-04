This project started as a project for CMSC 473, the Natural Language Processing class at UMBC. It compares two different Seq2Seq NLP models, which are supposed to take a Japanese sentence and give back the same sentence
with the kanji words converted to the correct phonetic readings in hiragana. Both are implemented using PyTorch,
and both are tokenized at the unicode codepoint level (so just Python characters). 
So far, both have been unsuccessful.

The training data is a file of about 245k Japanese sentence pairs with the input in the first column and the desired output in the second column.
The original sentence corpus was taken from [Tatoeba](http://tatoeba.org/), which is released under a [CC-BY License](http://creativecommons.org/licenses/by/2.0/fr/).
The output column was generated using Sudachi, a morphological analyzer and lattice-based tokenizer of Japanese that provides all the correct readings without machine learning.
