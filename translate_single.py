from transformers import T5Tokenizer, T5ForConditionalGeneration
import pysbd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model():
    model_name = "leukas/mt5-large-wmt14-1250k-deen"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def split_into_sentences(text, language="de"):
    seg = pysbd.Segmenter(language=language, clean=False)
    return seg.segment(text)

def translate_sentence(model, tokenizer, sentence):
    input_text = f"translate German to English: {sentence}"
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

def translate_long_text(model, tokenizer, text):
    sentences = split_into_sentences(text, language="de")
    translated_sentences = []

    for idx, sentence in enumerate(sentences, start=1):
        translated_sentence = translate_sentence(model, tokenizer, sentence)
        translated_sentences.append(translated_sentence)
        logging.info(f"Translated sentence {idx}/{len(sentences)}: {translated_sentence[:50]}...")

    return " ".join(translated_sentences)

def translate_file(input_file, output_file, model, tokenizer):
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            lines = infile.readlines()

        total_lines = len(lines)

        with open(output_file, "w", encoding="utf-8") as outfile:
            for line_idx, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                translated_text = translate_long_text(model, tokenizer, line)
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
