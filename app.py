from fastapi import FastAPI, Request
from model_loader import load_model
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

app = FastAPI()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@app.get("/embedding")
def get_embedding(text: str):
    model, tokenizer, device = load_model()
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding[0].cpu().tolist()
