from transformers import AutoTokenizer
import pandas as pd

def load_tokenizer(model_name="distilgpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_text(tokenizer, text: str):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    return tokens, token_ids

def tokens_to_dataframe(tokens, token_ids):
    clean_tokens = [token.replace("Ġ", " ") for token in tokens]
    df = pd.DataFrame({
        "index": list(range(len(tokens))),
        "token": tokens, 
        "clean_token": clean_tokens,
        "token_id": token_ids
    })
    return df