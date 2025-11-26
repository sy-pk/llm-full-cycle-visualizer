import streamlit as st

st.title("LLM Full Cycle Visualizer")

text = st.text_input("Enter a sentence: ")
if text:
    st.write("You entered:", text)