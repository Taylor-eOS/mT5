from transformers import T5Tokenizer, T5ForConditionalGeneration
import pysbd
from typing import List  # Fix for List type hint
import logging
import re
import spacy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load English NLP model for post-processing
nlp_en = spacy.load("en_core_web_sm")

def load_model():
    model_name = "leukas/mt5-large-wmt14-1250k-deen"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def split_into_sentences(text: str, language: str = "de") -> List[str]:
    """Robust sentence splitting using pysbd with language support"""
    segmenter = pysbd.Segmenter(language=language, clean=False)
    return list(segmenter.segment(text))

def calculate_token_length(text: str, tokenizer) -> int:
    """Accurate token length calculation including special tokens"""
    return len(tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0])

def split_into_chunks(sentences: List[str], tokenizer, max_length: int = 512, context_size: int = 1) -> List[str]:
    """
    Split sentences into chunks without overlapping, but include context from previous sentences.
    Each chunk will contain up to `max_length` tokens, including context.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for i, sentence in enumerate(sentences):
        sentence_length = calculate_token_length(sentence, tokenizer)

        # Add context from previous sentences
        context_sentences = sentences[max(0, i - context_size):i]
        context_length = sum(calculate_token_length(s, tokenizer) for s in context_sentences)

        # If adding this sentence exceeds the max length, finalize the current chunk
        if current_length + sentence_length + context_length > max_length:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        # Add context and current sentence to the chunk
        current_chunk.extend(context_sentences)
        current_chunk.append(sentence)
        current_length += context_length + sentence_length

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def translate_chunk(model, tokenizer, chunk: str) -> str:
    """Translate chunk with enhanced decoding parameters"""
    input_text = f"translate German to English: {chunk}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        num_beams=5,
        max_length=256,
        early_stopping=True,
        length_penalty=1.5,
        repetition_penalty=2.0,
        no_repeat_ngram_size=3,
        temperature=0.5,
        do_sample=False
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def postprocess_translation(text: str) -> str:
    """Fix punctuation and sentence boundaries using regex and spaCy"""
    # First pass: Regex-based fixes
    text = re.sub(r',(\s*)([“”"]?)(\s*)([A-Z0-9])', r'.\1\2\3\4', text)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    # Second pass: NLP-based correction
    doc = nlp_en(text)
    corrected = []
    for sent in doc.sents:
        txt = sent.text
        # Ensure sentence ends with proper punctuation
        if txt[-1] not in {'.', '!', '?'}:
            txt = txt.rstrip(',') + '.'
        corrected.append(txt)
    
    return ' '.join(corrected)

def translate_long_text(model, tokenizer, text: str, max_length: int = 512) -> str:
    """Enhanced translation pipeline with context handling"""
    # Step 1: Accurate sentence splitting
    sentences = split_into_sentences(text)
    
    # Step 2: Context-aware chunking
    chunks = split_into_chunks(sentences, tokenizer, max_length=max_length)
    
    # Step 3: Translate each chunk
    translated_chunks = []
    for chunk in chunks:
        translated = translate_chunk(model, tokenizer, chunk)
        translated_chunks.append(translated)
        logging.info(f"Translated chunk: {translated[:60]}...")

    # Step 4: Combine and post-process
    full_translation = " ".join(translated_chunks)
    return postprocess_translation(full_translation)

def translate_file(input_file, output_file, model, tokenizer):
    """Handle file translation with error checking"""
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            text = infile.read()

        # Process in meaningful segments (paragraphs)
        paragraphs = text.split("\n\n")
        translated_paragraphs = []
        
        for para in paragraphs:
            if para.strip():
                translated = translate_long_text(model, tokenizer, para)
                translated_paragraphs.append(translated)
                logging.info(f"Translated paragraph: {translated[:60]}...")
            else:
                translated_paragraphs.append("")

        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write("\n\n".join(translated_paragraphs))

        logging.info(f"Translation complete! Results saved in '{output_file}'")

    except Exception as e:
        logging.error(f"Error in translation: {str(e)}")
        raise

def main():
    input_file = "input.txt"
    output_file = "output.txt"
    
    logging.info("Loading models...")
    model, tokenizer = load_model()
    
    logging.info("Starting translation process...")
    try:
        translate_file(input_file, output_file, model, tokenizer)
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")

if __name__ == "__main__":
    main()
