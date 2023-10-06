import datasets as hds
from torch.utils.data import Dataset

hds.disable_progress_bar()


class HFDataset(Dataset):
    def __init__(self, ds: hds.Dataset) -> None:
        super().__init__()
        self.ds = ds

    def __getitem__(self, index):
        rel_cols = ["input_ids", "attention_mask", "labels"]
        ret_dict = {
            col: self.ds[col][index] for col in rel_cols if col in self.ds.column_names
        }
        return ret_dict

    def __len__(self) -> int:
        return len(self.ds)


class ContrastiveHFDataset(Dataset):
    def __init__(self, ds: HFDataset, candidates) -> None:
        super().__init__()
        self.ds = ds
        self.candidates = candidates

    def __getitem__(self, index):
        ret_dict = self.ds[index]
        candidate_data = self.ds[self.candidates[index]]
        candidate_data = {f"candidate_{k}": v for k, v in candidate_data.items()}

        return {**ret_dict, **candidate_data}

    def __len__(self) -> int:
        return len(self.ds)
