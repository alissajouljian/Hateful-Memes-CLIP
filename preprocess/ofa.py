import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from fairseq import checkpoint_utils
from fairseq import utils, tasks
from tasks.mm_tasks.caption import CaptionTask
from torchvision import transforms
from tqdm.auto import tqdm
from utils.eval_utils import eval_step


def generate_captions(json_dir, img_folder, save_csv, ofa_checkpoint):
    splits = ['train', 'dev', 'test']
    df = []
    for split_name in splits:
        file_path = os.path.join(json_dir, f'{split_name}.json')
        split_df = pd.read_json(file_path, lines=True)
        split_df['split'] = split_name
        df.append(split_df)

    df = pd.concat(df, axis=0, ignore_index=True)
    df['id'] = df['img'].str.split('/').str[1].str.split('.').str[0]
    df.index = df['id']
    df.index.name = None

    tasks.register_task('caption', CaptionTask)

    use_cuda = torch.cuda.is_available()
    use_fp16 = True

    overrides = {"bpe_dir": "utils/BPE", "eval_cider": False, "beam": 5, "max_len_b": 16, "no_repeat_ngram_size": 3,
                 "seed": 7}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(ofa_checkpoint),
        arg_overrides=overrides
    )

    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
            print('GPU is used')
        model.prepare_for_inference_(cfg)

    generator = task.build_generator(models, cfg.generation)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()

    def encode_text(text, length=None, append_bos=False, append_eos=False):
        s = task.tgt_dict.encode_line(
            line=task.bpe.encode(text),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([bos_item, s])
        if append_eos:
            s = torch.cat([s, eos_item])
        return s

    def construct_sample(image: Image):
        patch_image = patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        src_text = encode_text("explain, what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
        sample = {
            "id": np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask
            }
        }
        return sample

    def apply_half(t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    captions = []
    for img_fn in tqdm(df['img'].str.split('/').str[1]):
        img_fp = os.path.join(img_folder, img_fn)
        img = Image.open(img_fp)
        sample = construct_sample(img)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
        with torch.no_grad():
            result, scores = eval_step(task, generator, models, sample)
        captions.append(result[0]['caption'])

    df['caption'] = captions
    float_cols = df.select_dtypes(float).columns
    df[float_cols] = df.select_dtypes(float).astype('Int64')
    df.to_csv(save_csv, index=False)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate captions for images.')
    parser.add_argument('--json_dir', type=str, required=True, help='Directory for JSON labels')
    parser.add_argument('--img_folder', type=str, required=True, help='Folder containing images')
    parser.add_argument('--saved_csv_path', type=str, required=True, help='Path to save the processed CSV file')
    parser.add_argument('--ofa_checkpoint', type=str, required=True, help='Path to the model checkpoint file')

    args = parser.parse_args()

    generate_captions(args.json_dir, args.img_folder, args.saved_csv_path, args.ofa_checkpoint)
