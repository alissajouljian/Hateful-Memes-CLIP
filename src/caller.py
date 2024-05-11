import torch
from transformers import CLIPProcessor, CLIPTokenizer


class Caller(object):
    """
    A class to preprocess images and texts using CLIP models for a given batch
    """

    def __init__(self, args):
        """
        Initialize the Caller Class

        Parameters
        ----------
        args : argparse.Namespace
            Arguments passed to the Caller class

        Returns
        -------
        out : None
        """
        self.args = args
        self.image_processor = CLIPProcessor.from_pretrained(args.clip_pretrained_model)
        self.text_processor = CLIPTokenizer.from_pretrained(args.clip_pretrained_model)

    def __call__(self, batch):
        """
        Processes a batch of data dictionaries containing images, texts, captions, and labels

        Parameters
        ----------
        batch : List[Dict[str, Any]]
             A list of dictionaries each containing 'image', 'text', 'caption', and 'label'

        Returns
        -------
        out : Dict[str, torch.Tensor]
        """
        pixel_values = self.image_processor(images=[item['image'] for item in batch], return_tensors="pt")[
            'pixel_values']

        text_output = self.text_processor([item['text'] + ' [SEP] ' + item['caption'] for item in batch], padding=True,
                                          return_tensors="pt", truncation=True)

        caption_output = self.text_processor([item['caption'] for item in batch], padding=True, return_tensors="pt",
                                             truncation=True)
        labels = torch.LongTensor([item['label'] for item in batch])

        new_batch = {}
        new_batch['pixel_values'] = pixel_values
        new_batch['input_ids'] = text_output['input_ids']
        new_batch['attention_mask'] = text_output['attention_mask']

        new_batch['input_ids_caption'] = caption_output['input_ids']
        new_batch['attention_mask_caption'] = caption_output['attention_mask']
        new_batch['labels'] = labels

        return new_batch
