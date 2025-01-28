from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_model():
    model_name = "leukas/mt5-base-nc16-250k-deen"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

# Split long text into smaller overlapping chunks
def split_into_chunks(text, tokenizer, max_length=512, overlap=50):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        start = end - overlap  # Create overlap for continuity
    return chunks

# Translate a single chunk of tokens
def translate_chunk(model, tokenizer, chunk):
    input_text = tokenizer.decode(chunk, skip_special_tokens=True)
    input_text = f"translate German to English: {input_text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs, num_beams=5, max_length=512, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Translate a long text by splitting into chunks and combining results
def translate_long_text(model, tokenizer, text, max_length=512, overlap=50):
    chunks = split_into_chunks(text, tokenizer, max_length=max_length, overlap=overlap)
    translated_chunks = [translate_chunk(model, tokenizer, chunk) for chunk in chunks]
    return " ".join(translated_chunks)

# Process file line-by-line with chunk-based translation
def translate_file(input_file, output_file, model, tokenizer, max_length=512, overlap=50):
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            lines = infile.readlines()

        translated_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            translated_text = translate_long_text(model, tokenizer, line, max_length=max_length, overlap=overlap)
            translated_lines.append(translated_text)

        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write("\n".join(translated_lines))

        print(f"Translation complete! Results saved in '{output_file}'")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main function
def main():
    input_file = "input.txt"  # Replace with your input file path
    output_file = "output.txt"  # Replace with your output file path

    print("Loading fine-tuned mT5 model...")
    model, tokenizer = load_model()

    print(f"Translating text from '{input_file}' to '{output_file}'...")
    translate_file(input_file, output_file, model, tokenizer)

if __name__ == "__main__":
    main()

