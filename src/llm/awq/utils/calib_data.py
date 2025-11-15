import torch
from datasets import load_dataset


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=128, seq_len = 512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        dataset = dataset['train']
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > seq_len:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seq_len
    return [cat_samples[:, i*seq_len:(i+1)*seq_len] for i in range(n_split)]
