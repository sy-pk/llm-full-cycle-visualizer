from transformers import AutoModelForCausalLM


def load_model(model_name="gpt2"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=None, low_cpu_mem_usage=False
    )
    model.to("cpu")
    model.eval()
    return model


def get_token_embedding(model, token_id):
    return model.transformer.wte.weight[token_id].detach().cpu().numpy()


def get_position_embedding(model, position_id):
    return model.transformer.wpe.weight[position_id].detach().cpu().numpy()
