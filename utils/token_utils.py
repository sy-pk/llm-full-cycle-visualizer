from transformers import AutoTokenizer

def load_tokenizer(model_name="distilgpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_text(tokenizer, text: str):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    return tokens, token_ids