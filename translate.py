from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk import sent_tokenize
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model():
    model_name = "leukas/mt5-large-wmt14-1250k-deen"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def split_into_sentences(text, language="german"):
    """Split text into sentences using NLTK."""
    return sent_tokenize(text, language=language)

def split_into_chunks(sentences, tokenizer, max_length=512, overlap=2):
    """Split sentences into chunks with overlap, ensuring no sentence is split."""
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_tokens = tokenizer(sentence, return_tensors="pt", truncation=False)["input_ids"][0]
        sentence_length = len(sentence_tokens)

        if current_length + sentence_length > max_length:
            # Add the current chunk to the list
            chunks.append(" ".join(current_chunk))
            # Start a new chunk with overlap
            current_chunk = current_chunk[-overlap:]  # Carry over the last `overlap` sentences
            current_length = sum(len(tokenizer(s, return_tensors="pt", truncation=False)["input_ids"][0]) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def translate_chunk(model, tokenizer, chunk):
    input_text = f"translate German to English: {chunk}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(
        **inputs,
        num_beams=5,
        max_length=512,
        early_stopping=True,
        length_penalty=2.0,  # Encourage longer translations
        no_repeat_ngram_size=3  # Avoid repetition
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_long_text(model, tokenizer, text, max_length=512, overlap=2):
    sentences = split_into_sentences(text, language="german")
    chunks = split_into_chunks(sentences, tokenizer, max_length=max_length, overlap=overlap)
    translated_chunks = []

    for idx, chunk in enumerate(chunks, start=1):
        translated_chunk = translate_chunk(model, tokenizer, chunk)
        translated_chunks.append(translated_chunk)
        # Log progress and the last translated sentence
        logging.info(f"Translated chunk {idx}/{len(chunks)}: {translated_chunk[:50]}...")

    return " ".join(translated_chunks)

def translate_file(input_file, output_file, model, tokenizer, max_length=512, overlap=2):
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

                # Log progress for each line
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

