import streamlit as st
from utils.token_utils import load_tokenizer, tokenize_text, tokens_to_dataframe

st.title("LLM Full Cycle Visualizer")

# Load tokenizer
tokenizer = load_tokenizer()

text = st.text_input("Enter a sentence: ")

if text:
    # Tokenize the input text
    tokens, token_ids = tokenize_text(tokenizer, text)

    st.subheader("Tokens:")
    st.write(tokens)

    st.subheader("Token IDs:")
    st.write(token_ids)

    # Display token information in a table
    df = tokens_to_dataframe(tokens, token_ids)

    st.subheader("Token Information Table:")
    st.dataframe(df, use_container_width=True)