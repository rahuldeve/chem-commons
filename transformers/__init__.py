import pandas as pd
import datasets as hds

hds.disable_progress_bar()
from transformers import AutoTokenizer, AutoModel

from .models import ContrastiveModel
from .train import (
    embed_all,
    configure_optimizer_and_scheduler,
    prepare_dataloaders,
    train_one_epoch,
)
from .data import HFDataset
from .utils import TrainArgs


def load_tokenizer_and_model():
    tok = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    model: ContrastiveModel = ContrastiveModel(model)
    return model, tok


def get_transformer_embeddings(
    model: ContrastiveModel,
    tok: AutoTokenizer,
    smiles_data: hds.Dataset,
    smile_column_name,
):
    smiles_data = smiles_data.map(
        lambda entry: tok(
            entry[smile_column_name], padding="max_length", max_length=256
        )
    )
    smiles_data = smiles_data.remove_columns([smile_column_name])
    smiles_data.set_format("pt")

    embeddings = embed_all(model, HFDataset(smiles_data))
    return embeddings.detach().cpu().numpy()


def train(
    model: ContrastiveModel,
    tok: AutoTokenizer,
    data: hds.Dataset,
    smile_column_name,
    labels_column_name,
):
    data = data.map(
        lambda entry: tok(
            entry[smile_column_name], padding="max_length", max_length=128
        )
    )
    data = data.remove_columns([smile_column_name])
    data.set_format("pt")

    args = TrainArgs(seed=42, epochs=10, lr=2e-3)
    optimizer, scheduler, _, _ = configure_optimizer_and_scheduler(
        args, model, len(data)
    )

    for e in range(args.epochs):
        train_dl = prepare_dataloaders(model, data, args)
        train_one_epoch(model, train_dl, optimizer, scheduler, args, tok)

    return model
