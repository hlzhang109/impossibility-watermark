import torch
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class CustomSentenceTransformer:
    def __init__(self, model_name, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v1")
        self.model = AutoModel.from_pretrained(model_name)

        self.model.eval()
        self.model.to(self.device)

    def encode(self, sentences, batch_size=1, convert_to_tensor=False):
        embeddings = []
        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean pooling to get the sentence embedding
            embeddings = mean_pooling(outputs, inputs['attention_mask'])
            # TODO: Make sure these parameters are what you want. Maybe check in with Fabrice.
            # These parameters are from the HF page for "sentence-transformers/all-mpnet-base-v1". -- Boran
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # embeddings = torch.cat(embeddings, dim=0)

        return embeddings.cpu().numpy()