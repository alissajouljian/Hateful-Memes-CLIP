import argparse
import os

import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def generate_captions(json_dir, img_folder):
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

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

    captions = []
    for img_fn in df['img'].str.split('/').str[1]:
        img_fp = os.path.join(img_folder, img_fn)
        img = Image.open(img_fp).convert("RGB")
        device = "cuda"
        inputs = processor(images=img, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        captions.append(generated_caption)

    df['caption'] = captions
    float_cols = df.select_dtypes(include=[float]).columns
    df[float_cols] = df.select_dtypes(include=[float]).astype('Int64')
    df.to_csv("data.csv")


def main():
    parser = argparse.ArgumentParser(description="Generate captions for images in a dataset")
    parser.add_argument("--json_dir", type=str, default="/path/to/json/directory",
                        help="Directory where JSON files are stored")
    parser.add_argument("--img_folder", type=str, default="/path/to/image/folder",
                        help="Directory where images are stored")
    args = parser.parse_args()

    generate_captions(args.json_dir, args.img_folder)


if __name__ == "__main__":
    main()

