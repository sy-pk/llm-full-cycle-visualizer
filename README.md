# LLM Full Cycle Visualizer (Work In Process)

[![Streamlit Live Demo](https://img.shields.io/badge/Streamlit%20Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](https://sy-pk-llm-full-cycle-visualizer-app-gdqz1p.streamlit.app/)


This program is an educational tool designed to reveal the internal computation steps of transformer-based Large Language Models (LLMs).

The goal is to make the full LLM pipeline easier to understand by providing an “X-ray-style” view that allows users to follow each stage involved in generating a prediction.

Users can also experiment with sample inputs to manually trace through intermediate matrices and computations.

<br>

## What This Tool Visualizes

1. Tokenization — splitting text into byte-level BPE tokens  
2. Token → token ID conversion
3. Embedding lookup
4. Positional encoding
5. Transformer layers
   - Multi-head self-attention  
     - Construction of Q, K, V  
     - Attention score computation  
   - Feed-forward network (FFN)  
   - Residual connections + LayerNorm  
6. Final hidden state representation  
7. Logits computation
8. Softmax probability distribution
9. Selection of the next token    

<br>

## Technical Stack
- Model: DistilGPT-2 (lightweight and runs well on CPU)
- Framwork: Streamlit
- Visualization: Matplotlib / Seaborn

<br>

## How to Run 

#### 1. Clone to repository
```
https://github.com/sy-pk/llm-full-cycle-visualizer.git
```
```
cd llm-full-cycle-visualizer
```

#### 2. Create virtual environment
```
python3 -m venv .venv
```
```
source .venv/bin/activate
```

#### 3. Install dependencies
```
pip install -r requirements.txt
```

#### 4. Run the app 
```
streamlit run app.py
```
