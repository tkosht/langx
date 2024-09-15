import torch
import torch.nn as nn

from app.base.models.model import Classifier


class SimpleBertClassifier(Classifier):
    def __init__(
        self,
        bert,
        class_names: list[str],
        n_dim=768,
        n_hidden=128,
        droprate=0.5,
        weight=None,
    ) -> None:
        super().__init__(class_names)

        self.bert: nn.Module = bert
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.droprate = droprate
        self.weight = weight

        self.criterion = nn.CrossEntropyLoss(weight=weight)

        self.clf = nn.Sequential(
            nn.BatchNorm1d(self.n_dim),
            nn.Linear(self.n_dim, self.n_hidden),
            nn.BatchNorm1d(self.n_hidden),
            nn.GELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.n_hidden, self.n_classes),
            nn.BatchNorm1d(self.n_classes),
            nn.Softmax(dim=-1),
        )

        for p in self.bert.parameters():
            p.requires_grad = False

        for lyr in self.clf.parameters():
            if isinstance(lyr, nn.Linear):
                torch.nn.init.kaiming_uniform_(lyr.weight)

    def forward(self, *args, **kwargs):
        o = self.bert(*args, **kwargs)
        lh = o["last_hidden_state"]
        po = o["pooler_output"]
        h = lh.sum(axis=1)
        h = h + po
        y = self.clf(h)
        return y


class TransBertClassifier(Classifier):
    def __init__(
        self,
        bert,
        class_names: list[str],
        n_dim=768,
        n_hidden=128,
        droprate=0.5,
        weight=None,
    ) -> None:
        super().__init__(class_names)

        self.bert: nn.Module = bert
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.droprate = droprate
        self.weight = weight

        self.criterion = nn.CrossEntropyLoss(weight=weight)

        decoder_layer = nn.TransformerDecoderLayer(d_model=n_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.bn = nn.BatchNorm1d(self.n_dim)

        self.clf = nn.Sequential(
            # nn.BatchNorm1d(self.n_dim),
            self.transformer_decoder,
        )

        for p in self.bert.parameters():
            p.requires_grad = False

        for lyr in self.clf.parameters():
            if isinstance(lyr, nn.Linear):
                torch.nn.init.kaiming_uniform_(lyr.weight)

    def forward(self, *args, **kwargs):
        o = self.bert(*args, **kwargs)
        lh = o["last_hidden_state"]
        po = o["pooler_output"]

        mem = torch.transpose(lh, 0, 1)
        tgt = torch.zeros_like(mem)
        tgt[1:] = mem[:-1]
        dec = self.transformer_decoder(mem, tgt)
        dec = torch.transpose(dec, 0, 1)
        h = dec.sum(axis=1) + lh.sum(axis=1) + po

        y = self.clf(h)
        return y
