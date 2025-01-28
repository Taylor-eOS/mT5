from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from nltk.tokenize import sent_tokenize
import os

def load_model():
    model_name = "google/mt5-base"
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def read_input_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def write_output_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

def translate_text(model, tokenizer, text, max_length=512):
    # Prepare the input for translation
    input_text = f"translate German to English: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)
    # Generate the translation
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def main(input_file, output_file):
    print("Loading model...")
    model, tokenizer = load_model()
    print("Reading input file...")
    input_text = read_input_file(input_file)
    print("Splitting text into sentences...")
    sentences = sent_tokenize(input_text, language='german')
    print("Translating sentences...")
    translated_sentences = []
    for sentence in sentences:
        translated = translate_text(model, tokenizer, sentence)
        translated_sentences.append(translated)
    translated_text = " ".join(translated_sentences)

    print("Writing to output file...")
    write_output_file(output_file, translated_text)
    print(f"Translation complete! Output written to: {output_file}")

if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "output.txt"
    main(input_file, output_file)

