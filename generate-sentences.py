from sudachipy import tokenizer
from sudachipy import dictionary
from sudachipy import SplitMode

def convert_kanji_to_hiragana_with_spacing(text):
    tokenizer_obj = dictionary.Dictionary().create()
    morphemes = tokenizer_obj.tokenize(text, SplitMode.C)
    output_parts = []
    for m in morphemes:
        reading = m.reading_form()
        surface = m.surface()
        # Check if the surface contains kanji
        if any('\u4e00' <= char <= '\u9faf' for char in surface):
            # Convert katakana reading to hiragana
            hiragana_reading = "".join([chr(ord('ぁ') + ord(char) - ord('ァ'))
                                        if 'ァ' <= char <= 'ン'
                                        else char
                                        for char in reading])
            output_parts.append(hiragana_reading)
        else:
            output_parts.append(surface)
    return "".join(output_parts)

def process_file_sudachi(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        for line in infile:
            japanese_sentence = line.strip()
            converted_sentence = convert_kanji_to_hiragana_with_spacing(japanese_sentence)
            outfile.write(f"{japanese_sentence}\t{converted_sentence}\n")

if __name__ == "__main__":
    input_file = "jpn_sentences.txt"
    output_file = "kanji_hiragana_pairs.tsv"
    process_file_sudachi(input_file, output_file)
    print(f"Successfully processed '{input_file}' and saved the modified output to '{output_file}'.")
