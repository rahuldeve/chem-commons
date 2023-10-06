import torch.nn.functional as F
import torch.nn as nn
import torch
from enum import Enum


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class ContrastiveModel(nn.Module):
    modes = Enum("MODE", ["train", "infer"])

    def __init__(self, base_model) -> None:
        super().__init__()
        self.base_model = base_model

    def embed(self, outputs, attention_mask):
        # embeds = outputs.last_hidden_state[:, 0, :]
        embeds = mean_pooling(outputs, attention_mask)
        return embeds

    def train_mode(
        self,
        labels,
        input_ids,
        attention_mask,
        candidate_labels,
        candidate_input_ids,
        candidate_attention_mask,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        embeds = self.embed(outputs, attention_mask).unsqueeze(-2)

        B, C, L = candidate_attention_mask.shape
        candidate_input_ids = candidate_input_ids.view(-1, L)
        candidate_attention_mask = candidate_attention_mask.view(-1, L)
        outputs = self.base_model(
            input_ids=candidate_input_ids,
            attention_mask=candidate_attention_mask,
            output_hidden_states=True,
        )
        candidate_embeds = self.embed(outputs, candidate_attention_mask).view(B, C, -1)

        pred_logits = (embeds @ candidate_embeds.transpose(-1, -2)).squeeze()
        mask = labels.unsqueeze(-1) == candidate_labels

        target_contrib = F.logsigmoid(pred_logits[mask]).sum()
        noise_contrib = F.logsigmoid(-pred_logits[~mask]).sum()

        loss = -(target_contrib + noise_contrib) / B
        loss_target = -target_contrib / B
        loss_noise = -noise_contrib / B
        return (loss, loss_target, loss_noise, embeds)

    def infer_mode(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        embeds = self.embed(outputs, attention_mask)
        return embeds

    def forward(self, mode, **kwargs):
        if mode == ContrastiveModel.modes.train:
            return self.train_mode(**kwargs)
        elif mode == ContrastiveModel.modes.infer:
            return self.infer_mode(**kwargs)
        else:
            raise Exception("BAD mode provided")
