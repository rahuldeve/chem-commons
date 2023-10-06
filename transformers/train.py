import torch
import torch.nn.functional as F
import datasets as hds
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from .models import ContrastiveModel
from .data import HFDataset, ContrastiveHFDataset
from .utils import TrainArgs

from tqdm.auto import tqdm


@torch.no_grad()
def embed_all(model: ContrastiveModel, ds: HFDataset):
    model.eval()
    device = next(model.parameters()).device
    dl = DataLoader(ds, batch_size=256)
    all_embeds = []
    for batch in tqdm(dl, total=len(dl), disable=True):
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if k in {"input_ids", "attention_mask"}
        }
        embeds = model(ContrastiveModel.modes.infer, **batch)
        all_embeds.append(embeds)

    all_embeds = torch.cat(all_embeds, dim=0)
    return all_embeds.cpu()


def configure_optimizer_and_scheduler(args: TrainArgs, model, num_train_examples: int):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)

    num_train_steps = int(
        num_train_examples / args.B / args.gradient_accumulation_steps * args.epochs
    )
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    return optimizer, scheduler, num_train_steps, num_warmup_steps


@torch.no_grad()
def mine_candidates(all_embeds: torch.Tensor, args: TrainArgs):
    crosswise_pred_sims = all_embeds @ all_embeds.T
    # closest_pred_embeds = torch.topk(crosswise_pred_sims, dim=-1, k=args.n_candidates)
    # closest_pred_embeds_idxs = closest_pred_embeds.indices[:, :]
    # return closest_pred_embeds_idxs.cpu()

    sorted_pred_embeds_idxs = torch.argsort(crosswise_pred_sims, dim=-1)[:, 1:]
    n_hard = int(args.n_candidates * 0.6)
    hard_candidate_idxs = sorted_pred_embeds_idxs[:, :n_hard]
    n_random = int(args.n_candidates * 0.4)
    random_candidates_idxs = sorted_pred_embeds_idxs[
        :, n_hard + torch.randperm(n_random)
    ]
    candidates = torch.cat([hard_candidate_idxs, random_candidates_idxs], dim=-1)
    return candidates.cpu()


def prepare_dataloaders(model: ContrastiveModel, ds: hds.Dataset, args: TrainArgs):
    train_ds = HFDataset(ds)
    train_embeds = embed_all(model, train_ds)
    train_candidates = mine_candidates(train_embeds, args)
    train_dl = DataLoader(
        ContrastiveHFDataset(train_ds, train_candidates),
        shuffle=True,
        batch_size=args.B,
        drop_last=True,
    )
    return train_dl


def train_one_epoch(
    model: ContrastiveModel,
    train_dl: DataLoader,
    optimizer,
    scheduler,
    args: TrainArgs,
    tok,
):
    model.train()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_loss_target = 0.0
    total_loss_noise = 0.0

    for step, batch in enumerate(train_dl):
        # batch['input_ids'] = torch_mask_tokens(batch['input_ids'], tok)[0]
        optimizer.zero_grad()
        model.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        loss, loss_target, loss_noise, _ = model(ContrastiveModel.modes.train, **batch)
        loss.backward()
        total_loss += loss.cpu().item()
        total_loss_target += loss_target.cpu().item()
        total_loss_noise += loss_noise.cpu().item()

        if step % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

    N = len(train_dl)
    return (total_loss / N, total_loss_target / N, total_loss_noise / N)
