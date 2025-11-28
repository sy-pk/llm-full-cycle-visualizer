from transformers import AutoModelForCausalLM
import torch

def load_model(model_name="distilgpt2"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model

def get_token_embedding(model, token_id):
    return model.transformer.wte.weight[token_id].detach().cpu().numpy()

def get_position_embedding(model, position_id):
    return model.transformer.wpe.weight[position_id].detach().cpu().numpy()