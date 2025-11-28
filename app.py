import streamlit as st

from utils.model_utils import get_position_embedding, get_token_embedding, load_model
from utils.token_utils import load_tokenizer, tokenize_text, tokens_to_dataframe
from utils.visualization_utils import plot_vector_heatmap


def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def add_tooltip(text, tooltip_dictionary):
    for term, tip in tooltip_dictionary.items():
        text = text.replace(
            term,
            f'<span class="tooltip">{term}<span class="tooltiptext">{tip}</span></span>',
        )
    return text


def md(text):
    processed = add_tooltip(text, tooltip_dictionary)
    st.markdown(processed, unsafe_allow_html=True)


load_css("styles/tooltip.css")

tokenizer = load_tokenizer()
model = load_model()

tooltip_dictionary = {
    "LLM": "Large Language Model",
}

st.title("🧠 LLM Full Cycle Visualizer")

md(
    """
Ever wondered how AI models like ChatGPT or Gemini manage to respond so much like humans?

This application lets you peek inside LLM to see what happens between your input 
and the AI's answer, almost like an X-ray. 
"""
)

st.subheader("Input Text")
user_input_text = st.text_input("Enter a sentence: ")

if user_input_text:
    # Tokenize the input text
    tokens, token_ids = tokenize_text(tokenizer, user_input_text)

    st.subheader("1. Splitting into Tokens:")
    st.markdown(
        """A sentence like \"Hello world!\" looks simple to us, but for AI, 
    it must be broken into smaller pieces called tokens."""
    )
    st.write(tokens)

    # Get the vocabulary as a dictionary
    vocab = tokenizer.get_vocab()

    # Get the size of the vocabulary
    vocab_size = len(vocab)
    st.write(f"DistilGPT2's vocabulary size (same tokenizer as GPT-2): {vocab_size}")

    st.link_button("OpenAI Tokenizer", "https://platform.openai.com/tokenizer")

    st.subheader("2. Turning tokens into numbers (Token IDs):")
    st.write(token_ids)

    # Display token information in a table (index, token, clean_token, token_id)
    token_dataframe = tokens_to_dataframe(tokens, token_ids)

    st.subheader("Token Information Table:")
    st.dataframe(token_dataframe, use_container_width=True)

    # Select token for Embedding Visualization
    selected_token_index = st.slider(
        label="Select a token index:", min_value=0, max_value=len(tokens) - 1, value=0
    )

    selected_token = tokens[selected_token_index]
    selected_token_id = token_ids[selected_token_index]

    st.subheader(f"Selected Token: '{selected_token}' (ID: {selected_token_id})")

    # Extract Embeddings
    token_embedding_vector = get_token_embedding(model, selected_token_id)
    position_embedding_vector = get_position_embedding(model, selected_token_index)
    combined_embedding_vector = token_embedding_vector + position_embedding_vector

    # Visualize Embeddings as Heatmaps
    st.subheader("Token Embedding Vector:")
    st.pyplot(plot_vector_heatmap(token_embedding_vector, "Token Embedding"))

    st.subheader("Position Embedding Vector:")
    st.pyplot(plot_vector_heatmap(position_embedding_vector, "Position Embedding"))

    st.subheader("Combined Embedding (Token + Position):")
    st.pyplot(plot_vector_heatmap(combined_embedding_vector, "Combined Embedding"))

st.markdown(
    """
<hr style='margin-top: 60px; margin-bottom: 20px; border: 1px solid #444;'>
            
<details>
<summary>References</summary>

<br>

[1] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, 
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, 
Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, 
Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, 
Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, 
Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, 
Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are 
few-shot learners. In Proceedings of the 34th International Conference on Neural 
Information Processing Systems (NeurIPS ’20), December 6–12, 2020, Vancouver, Canada. 
Curran Associates Inc., New York, NY, 1877–1901.  
PDF: [NeurIPS Proceedings (2020)](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

</details>
""",
    unsafe_allow_html=True,
)
