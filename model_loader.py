from transformers import AutoTokenizer, AutoModel
import torch

model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
        model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
        model.to(device)
        model.eval()
    return model, tokenizer, device
