import torch
from datasets import load_dataset
from calibration.pileval import get_calib_dataset
from calibration.coco_vl import get_multimodal_calib_dataset

def get_lm_calib_dataset(data="pileval", tokenizer=None, n_samples=128, seq_len = 512):
    if data == "pileval":
        # dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        dataset = load_dataset("json", data_files="/home/lzc/workspace/datasets/pile-val-backup/val.jsonl")
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

def get_vlm_calib_dataset(
        data, data_path, image_folder, model,
        tokenizer, n_samples, few_shot_format,
        interleave_format, text_data_path
):
        # Generate the calibration tokens.
    prompt_inputs = None
    prompt_kwargs = None

    if data == "pileval":
        prompt_inputs, prompt_kwargs = get_calib_dataset(data_path=data_path, tokenizer=tokenizer, n_samples=n_samples)
    elif data == "coco":
        prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(data_path=data_path,
                                                                    image_folder=image_folder,
                                                                    model=model,
                                                                    n_samples=n_samples,
                                                                    few_shot_format=few_shot_format,
                                                                    interleave_format=interleave_format,
                                                                    text_data_path=text_data_path)
    return prompt_inputs, prompt_kwargs