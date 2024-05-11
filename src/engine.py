import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from transformers import CLIPModel


class TrainCLIP(pl.LightningModule):
    """
    A module that trains and evaluates CLIP model with a specific pipeline
    """

    def __init__(self, args):
        """
        Initialize the TrainCLIP Class for Training

        Parameters
        ----------
        args : argparse.Namespace
            Arguments passed to the TrainCLIP class

        Returns
        -------
        out : None
        """
        super().__init__()

        self.map_dim = args.map_dim
        self.num_pre_output_layers = args.num_pre_output_layers
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.acc = torchmetrics.Accuracy(task="binary", num_classes=1)

        self.auroc = torchmetrics.AUROC(task="binary", num_classes=1)
        self.precision_score = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")

        self.clip = CLIPModel.from_pretrained(args.clip_pretrained_model)

        self.image_encoder = copy.deepcopy(self.clip.vision_model)

        self.text_encoder = copy.deepcopy(self.clip.text_model)

        self.image_map = nn.Sequential(
            copy.deepcopy(self.clip.visual_projection),
            nn.ReLU(),
            nn.Linear(self.clip.projection_dim, self.map_dim)
        )
        self.text_map = nn.Sequential(
            copy.deepcopy(self.clip.text_projection),
            nn.ReLU(),
            nn.Linear(self.clip.projection_dim, self.map_dim)
        )

        pre_output_input_dim = self.map_dim

        pre_output_layers = [nn.Dropout(p=args.drop_probs[1])]
        output_input_dim = pre_output_input_dim
        if self.num_pre_output_layers >= 1:  # first pre-output layer
            pre_output_layers.extend(
                [nn.Linear(pre_output_input_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])
            output_input_dim = self.map_dim
        for _ in range(1, self.num_pre_output_layers):  # next pre-output layers
            pre_output_layers.extend(
                [nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.drop_probs[2])])

        self.pre_output = nn.Sequential(*pre_output_layers)

        self.output = nn.Linear(output_input_dim, 1)

        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        # frezze
        if True:
            for _, p in self.image_encoder.named_parameters():
                p.requires_grad_(False)

        if True:
            for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)

    def forward(self, batch):
        """
        Forward pass for generating predictions from the model

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Input batch

        Returns
        -------
        out : torch.Tensor
        """
        image_features = self.image_encoder(pixel_values=batch['pixel_values']).pooler_output
        image_features = self.image_map(image_features)

        text_features = self.text_encoder(input_ids=batch['input_ids'],
                                          attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)

        image_features = F.normalize(image_features, p=2, dim=1)  # Normalize the image features
        text_features = F.normalize(text_features, p=2, dim=1)  # Normalize the text features

        features = torch.mul(image_features, text_features)  # Element-wise multiplication of features

        features = self.pre_output(features)
        logits = self.output(features)
        preds = (torch.sigmoid(logits) >= 0.5).long()

        return preds

    def common_step(self, batch):
        """
        Compute the loss and update the metrics based on the predictions

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Input batch

        Returns
        -------
        out : Dict[str, torch.Tensor]
        """
        image_features = self.image_encoder(pixel_values=batch['pixel_values']).pooler_output
        image_features = self.image_map(image_features)
        text_features = self.text_encoder(input_ids=batch['input_ids'],
                                          attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        output = {}
        features = torch.mul(image_features, text_features)

        features_pre_output = self.pre_output(features)
        logits = self.output(features_pre_output).squeeze(dim=1)

        preds_proxy = torch.sigmoid(logits)
        preds = (preds_proxy >= 0.5).long()

        output['loss'] = self.cross_entropy_loss(logits, batch['labels'].float())
        output['accuracy'] = self.acc(preds, batch['labels'])
        output['auroc'] = self.auroc(preds_proxy, batch['labels'])

        return output

    def training_step(self, batch, batch_idx):
        """
        Performs one training step

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Input batch

        Returns
        -------
        out : torch.Tensor
        """
        output = self.common_step(batch)

        total_loss = output['loss']

        self.log('train/total_loss', total_loss)
        self.log('train/loss', output['loss'])
        self.log('train/accuracy', output['accuracy'])
        self.log('train/auroc', output['auroc'])

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Performs one validation step

        Parameters
        ----------
        batch : Dict[str, Tensor]
            The batch data containing inputs and labels.

        Returns
        -------
        out : torch.Tensor
        """
        output = self.common_step(batch)

        total_loss = output['loss']

        self.log(f'val/total_loss', total_loss)
        self.log(f'val/loss', output['loss'])
        self.log(f'val/accuracy', output['accuracy'])
        self.log(f'val/auroc', output['auroc'])

        return total_loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        """
        Performs test step

        Parameters
        ----------
        batch : Dict[str, Tensor]
            The batch data containing inputs and labels.

        Returns
        -------
        out : Dict[str, torch.Tensor]
        """
        prefix_map = {
            0: 'dev',
            1: 'test'
        }
        prefix = prefix_map[dataloader_idx]
        if dataloader_idx == 0:
            calling_function = 'validation'
        elif dataloader_idx == 1:
            calling_function = 'training'

        output = self.common_step(batch)

        self.log(f'{prefix}/accuracy', output['accuracy'])
        self.log(f'{prefix}/auroc', output['auroc'])

        return output

    def on_train_epoch_end(self):
        """
        Performs test step
        """
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def on_validation_epoch_end(self):
        """
        Count metrics on Validation epoch end
        """
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def on_test_epoch_end(self):
        """
        Count metrics on Test epoch end
        """
        self.acc.reset()
        self.auroc.reset()
        self.precision_score.reset()
        self.recall.reset()
        self.f1.reset()

    def configure_optimizers(self):
        """
        Sets up the optimizer for the model
        """
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
