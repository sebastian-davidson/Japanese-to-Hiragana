import csv
from sudachipy import tokenizer
from sudachipy import dictionary

tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

def to_kana(sentence):
    kana = ''.join([token.reading_form() for token in tokenizer_obj.tokenize(sentence, mode)])
    return katakana_to_hiragana(kana)

def katakana_to_hiragana(text):
    return text.translate(str.maketrans(
        "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホ"
        "マミムメモヤユヨラリルレロワヲンァィゥェォッャュョ",
        "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほ"
        "まみむめもやゆよらりるれろわをんぁぃぅぇぉっゃゅょ"
    ))

# Load sentences from the Tatoeba sentence file
pairs = []
with open("jpn_sentences.txt", encoding="utf-8") as f:
    # reader = csv.reader(f, delimiter='\\t')
    # jpn_sentences = [row[2] for row in reader if row[1] == 'jpn' and len(row[2]) < 100]
    for sentence in f:
        try:
            reading = to_kana(sentence)
            pairs.append((sentence.strip(),reading))
        except Exception as e:
            print(e)
            continue

"""
# Generate (kanji, hiragana) pairs
pairs = []
for s in jpn_sentences:
    try:
        reading = to_kana(s)
        pairs.append((s, reading))
    except Exception as e:
        continue
"""

# Save to file
with open("kanji_hiragana_pairs.tsv", "w", encoding="utf-8") as f:
    for orig, kana in pairs:
        f.write(f"{orig}\t{kana}")

