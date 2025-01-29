import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk import sent_tokenize
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model():
    model_name = "leukas/mt5-large-wmt14-250k-deen"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def preprocess_text(text):
    """Replace sentence-ending periods with a unique token to preserve sentence boundaries."""
    # Use regex to replace periods that are likely sentence endings
    text = re.sub(r'(?<=[.!?])\s+(?=[A-Z])', " [PERIOD] ", text)
    return text

def postprocess_text(text):
    """Replace the unique token back with periods."""
    text = re.sub(r'\s*\[PERIOD\]\s*', ". ", text).strip()
    return text

def enforce_capitalization(text):
    return re.sub(r'(?<=[.!?]\s)([a-z])', lambda m: m.group(1).upper(), text)

def split_into_sentences(text, language="german"):
    """Split text into sentences using NLTK."""
    return sent_tokenize(text, language=language)

def split_into_chunks(sentences, tokenizer, max_length=512, overlap=2):
    """Split sentences into chunks that fit within the model's max token limit."""
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_tokens = tokenizer(sentence, return_tensors="pt", truncation=False)["input_ids"][0]
        sentence_length = len(sentence_tokens)
        if current_length + sentence_length > max_length:
            chunks.append(" ".join(current_chunk))
            overlap_start = max(0, len(current_chunk) - overlap)
            current_chunk = current_chunk[overlap_start:]
            current_length = sum(len(tokenizer(s, return_tensors="pt", truncation=False)["input_ids"][0]) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def translate_chunk(model, tokenizer, chunk):
    """Translate a single chunk of text."""
    input_text = f"translate German to English: {chunk}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(
        **inputs,
        num_beams=5,
        max_length=512,
        early_stopping=True,
        length_penalty=2.0,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_long_text(model, tokenizer, text, max_length=512, overlap=2):
    """Translate long text by splitting it into manageable chunks."""
    # Preprocess text to preserve periods
    text = preprocess_text(text)
    sentences = split_into_sentences(text, language="german")
    chunks = split_into_chunks(sentences, tokenizer, max_length=max_length, overlap=overlap)
    translated_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        translated_chunk = translate_chunk(model, tokenizer, chunk)
        translated_chunks.append(translated_chunk)
        logging.info(f"Translated chunk {idx}/{len(chunks)}: {translated_chunk[:50]}...")
    
    # Postprocess translated text to restore periods
    translated_text = " ".join(translated_chunks)
    translated_text = postprocess_text(translated_text)
    translated_text = enforce_capitalization(translated_text)
    return translated_text

def translate_file(input_file, output_file, model, tokenizer, max_length=512, overlap=2):
    """Translate text from an input file and save results to an output file."""
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            lines = infile.readlines()
        total_lines = len(lines)
        with open(output_file, "w", encoding="utf-8") as outfile:
            for line_idx, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                translated_text = translate_long_text(model, tokenizer, line, max_length=max_length, overlap=overlap)
                outfile.write(translated_text + "\n")
                logging.info(f"Processed line {line_idx}/{total_lines}: {translated_text[:50]}...")
        logging.info(f"Translation complete! Results saved in '{output_file}'")
    except FileNotFoundError:
        logging.error(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def main():
    input_file = "input.txt"
    output_file = "output.txt"
    logging.info("Loading fine-tuned mT5 model...")
    model, tokenizer = load_model()
    logging.info(f"Translating text from '{input_file}' to '{output_file}'...")
    translate_file(input_file, output_file, model, tokenizer)

if __name__ == "__main__":
    main()

