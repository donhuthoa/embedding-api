from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

app = FastAPI()

class TextRequest(BaseModel):
    text: str

# Load tokenizer và model 1 lần duy nhất
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L3-v2")

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@app.post("/embed")
async def get_embedding(req: TextRequest):
    encoded_input = tokenizer(req.text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    embedding = F.normalize(embedding, p=2, dim=1)
    embedding_list = embedding[0].cpu().tolist()
    return {"embedding": embedding_list}
