import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from typing_extensions import Self

from app.base.component.params import add_args
from app.base.models.model import Classifier
from app.general.models.positional_encoding import PositionalEncoding


class BertClassifier(Classifier):
    @add_args(params_file="conf/app.yml", root_key="/model/transformer")
    def __init__(
        self,
        bert,
        class_names: list[str],
        n_dim=768,
        n_hidden=128,
        n_out=None,
        droprate=0.5,
        weight=None,
        params_encoder: DictConfig = None,
        params_decoder: DictConfig = None,
    ) -> None:
        super().__init__(class_names)

        self.bert: nn.Module = bert
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.n_out = len(class_names) if n_out is None else n_out
        self.droprate = droprate
        self.weight = weight
        self.params_encoder = params_encoder
        self.params_decoder = params_decoder
        self.pe_max_len = 1000
        self.step = 0

        self.W = self.bert.embeddings.word_embeddings.weight  # (V, D)
        self.pe = PositionalEncoding(
            self.n_dim, dropout=droprate, max_len=self.pe_max_len
        )
        # TODO: use Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_dim, nhead=params_encoder.nhead
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=params_encoder.num_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_dim, nhead=params_decoder.nhead
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=params_decoder.num_layers
        )

        self.clf = nn.Sequential(
            # nn.Linear(self.n_dim, self.n_out),
            nn.BatchNorm1d(self.n_out),
            nn.LogSoftmax(dim=-1),
        )

        # loss
        self.cel = nn.CrossEntropyLoss(weight=weight)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.mse = nn.MSELoss()
        self.kld = nn.KLDivLoss(reduction="batchmean")

        self.device = torch.device("cpu")

        self._initialize()

    def to(self, obj):
        super().to(obj)
        if isinstance(obj, torch.device):
            self.device = obj

    def _initialize(self) -> Self:
        for p in self.bert.parameters():
            p.requires_grad = False

        for lyr in self.parameters():
            if isinstance(lyr, nn.Linear):
                torch.nn.init.kaiming_uniform_(lyr.weight)

        return self

    def create_unk_for(self, mem: torch.Tensor):
        # mem : (S, B, D)
        V, D = self.W.shape

        # overwrite by unk vector
        S, B, D = mem.shape
        tokenizer = self.context["tokenizer"]
        unk_idx = tokenizer.unk_token_id
        unk = (
            F.one_hot(torch.LongTensor([unk_idx]), num_classes=V)
            .to(torch.float32)
            .reshape(1, 1, -1)
            .repeat(1, B, 1)
            .to(self.device)
        )
        unk_vector = torch.matmul(unk, self.W)
        U = torch.zeros_like(mem)
        noise_idx = torch.randint(0, S, (1,)).item()
        U[noise_idx] = unk_vector - mem.detach()[noise_idx]

        return U

    def create_right_shift_target(self, T: torch.Tensor):
        shp = list(T.shape)
        tgt = torch.zeros(shp, device=T.device)
        tgt[1:] = T[:-1]
        return tgt

    def create_mask_for(self, tgt: torch.Tensor):
        # tgt: (S, B, *)
        seq_len = tgt.shape[0]
        mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=self.device
        )
        return mask

    def embed(self, onehot: torch.Tensor, context_key=None):
        # onehot : (B, S', V)
        _emb = torch.matmul(onehot, self.W)  # (B, S', D)
        _emb = self.pe(_emb)
        emb = torch.transpose(_emb, 0, 1)  # -> (S', B, D)
        if context_key:
            self.context[context_key] = _emb
        return emb  # (S', B, D)

    def deembed(self, emb_seq: torch.Tensor, context_key=None):
        # emb_seq : (S', B, D)
        _emb = torch.transpose(emb_seq, 0, 1)  # -> (B, S', D)
        dembed = torch.matmul(_emb, self.W.T)  # (B, S', V)
        if context_key:
            self.context[context_key] = _emb
        return dembed  # (B, S', V)

    def _infer_decoding(
        self, tgt: torch.Tensor, mem: torch.Tensor, add_noise: bool = False
    ):
        # NOTE: 直接y の結果をtgt_ids の代わりに渡すことで、学習できるようにする

        if add_noise:
            # add unk tensor (more exactly, replace unk vectors)
            U = self.create_unk_for(mem)
            mem = mem + U

            # add noise
            D = mem.shape[-1]
            N = torch.normal(0, 1e-3 / D, mem.shape).to(mem.device)
            mem = mem + N

        tgt_msk = self.create_mask_for(tgt)
        assert tgt_msk.shape[0] == tgt.shape[0]
        dec = self.decoder(tgt, mem, tgt_mask=tgt_msk)  # -> (S, B, D)
        h = self.deembed(dec, "dec")  # -> (B, S, V)
        B, S, V = h.shape
        h = h.reshape(-1, V)  # -> (B*S, V)
        # h = torch.transpose(dec, 0, 1)  # -> (B, S, D)
        # B, S, D = h.shape
        # h = h.reshape(-1, D)  # -> (B*S, D)
        y = self.clf(h).reshape(B, S, -1)
        return y

    def forward(self, *args, **kwargs) -> torch.Tensor:
        o = self.bert(*args, **kwargs)
        h = torch.transpose(o["last_hidden_state"], 0, 1)  # -> (S, B, D)

        mem = self.encoder(h)  # (S, B, D)

        tokenizer = self.context["tokenizer"]
        tgt_ids = self.context["tgt_ids"]  # (B, S')
        tgt_onehot = F.one_hot(tgt_ids, tokenizer.vocab_size)  # (B, S) -> (B, S, V)
        tgt = self.embed(
            tgt_onehot.to(torch.float32).to(self.device), "trg"
        )  # -> (S, B, V)
        y = self._infer_decoding(tgt, mem, add_noise=self.params_decoder.add_noise)

        return y

    # def forward(self, *args, **kwargs) -> torch.Tensor:
    #     if self.training:
    #         self.step += 1
    #     if (
    #         self.step <= self.params_decoder.warmup_steps
    #         or torch.rand((1,)).item() < 0.5
    #     ):
    #         y = self._forward0(*args, **kwargs)
    #     else:
    #         # NOTE: warmup_steps 以降、0.5 の確率で eval と同じforward ステップをふむ
    #         y = self._forward1(*args, **kwargs)
    #     # y = self._forward0(*args, **kwargs)
    #     return y

    # def _forward1(self, *args, **kwargs) -> torch.Tensor:
    #     o = self.bert(*args, **kwargs)
    #     h = torch.transpose(o["last_hidden_state"], 0, 1)  # -> (S, B, D)

    #     mem = self.encoder(h)  # (S, B, D)

    #     tokenizer = self.context["tokenizer"]

    #     # setup tgt
    #     tgt_ids = self.context["tgt_ids"]  # (B, S')
    #     B, tgt_seqlen = tgt_ids.shape[:2]
    #     tgt_ids = (
    #         torch.LongTensor([tokenizer.cls_token_id])
    #         .unsqueeze(0)
    #         .repeat(B, 1)
    #         .to(self.device)
    #     )
    #     tgt_onehot = F.one_hot(tgt_ids, tokenizer.vocab_size)  # (B, S) -> (B, S, V)
    #     tgt_onehot = tgt_onehot.to(torch.float32).to(self.device)
    #     _tgt = tgt_onehot  # (B, S, V)

    #     for sdx in range(tgt_seqlen):
    #         tgt = self.embed(_tgt)  # -> (S, B, D)
    #         y = self._infer_decoding(tgt, mem)  # -> (B, S, V)
    #         y = torch.exp(y)  # LogSoftmax -> Softmax   # NOTE: argmax とるなら、なくてもよい
    #         # y = F.one_hot(y.argmax(dim=-1).long(), tokenizer.vocab_size)
    #         _tgt = torch.cat([tgt_onehot[:, :1], y], dim=1)  # -> (B, S+1, V)

    #     assert y.shape[1] == tgt_seqlen
    #     return y

    def predict(self, *args, **kwargs) -> torch.Tensor:
        o = self.bert(*args, **kwargs)
        h = torch.transpose(o["last_hidden_state"], 0, 1)  # -> (S, B, D)
        # mem = self.encoder(h)  # (S, B, D)
        # po = o["pooler_output"]
        mem = h  # (S, B, D)

        tokenizer = self.context["tokenizer"]

        # setup tgt
        B = mem.shape[1]
        max_seqlen = 8
        tgt_ids = (
            torch.LongTensor([tokenizer.cls_token_id])
            .unsqueeze(0)
            .repeat(B, 1)
            .to(self.device)
        )
        tgt_onehot = F.one_hot(tgt_ids, tokenizer.vocab_size)  # (B, S) -> (B, S, V)
        tgt_onehot = tgt_onehot.to(torch.float32).to(self.device)
        _tgt = tgt_onehot
        # zeros = torch.zeros_like(tgt_onehot).repeat(1, max_seqlen - 2, 1)
        # _tgt = torch.cat([tgt_onehot, zeros], dim=1)  # (B, S, V)

        # TODO: 長さをmax_seqlen -1 に合わせる
        #       パディングonehotの代わりに、zero ベクトルを使う
        for sdx in range(max_seqlen - 1):
            tgt = self.embed(_tgt)  # -> (S, B, D)
            # zero ベクトルをcat で追加
            y = self._infer_decoding(tgt, mem)  # -> (B, S, V)
            y = torch.exp(y)  # LogSoftmax -> Softmax   # NOTE: argmax とるなら、なくてもよい
            y = F.one_hot(y.argmax(dim=-1).long(), tokenizer.vocab_size)
            _tgt = torch.cat([tgt_onehot[:, :1], y], dim=1)  # -> (B, S+1, V)

        return y

    def to_text(self, y_rec: torch.Tensor, do_argmax=True) -> torch.Tensor:
        tokenizer = self.context["tokenizer"]
        y = y_rec.argmax(dim=-1) if do_argmax else y_rec
        return tokenizer.decode(y)

    def to_tokens(self, y_rec: torch.Tensor, do_argmax=True) -> list[str]:
        tokenizer = self.context["tokenizer"]
        y = y_rec.argmax(dim=-1) if do_argmax else y_rec
        return tokenizer.convert_ids_to_tokens(y)

    def loss(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # loss = self._loss_end(y, t) + self._loss_middle()
        # loss = self._loss_end(y, t)
        # loss = self._loss_seq(y, t)
        # loss = self.kld(y, t)
        loss = super().loss(y, t)
        return loss

    def _loss_seq(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = y.shape[0]
        _y = y.reshape((B, -1))  # -> (B, *)
        _t = t.reshape((B, -1))  # -> (B, *)
        loss_seq = super().loss(_y, _t) + self._loss_difference(_y, _t)
        # loss_seq = super().loss(_y, _t)
        return loss_seq

    def _loss_end(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = y.shape[0]
        _y = y.reshape((B, -1))  # -> (B, *)
        _t = t.reshape((B, -1))  # -> (B, *)
        loss_token = super().loss(y, t) + self._loss_difference(y, t)
        loss_seq = super().loss(_y, _t) + self._loss_difference(_y, _t)
        loss = loss_token + loss_seq
        return loss

    def _loss_middle(self):
        dec: torch.Tensor = self.context["dec"]  # -> (B, S', D)
        trg: torch.Tensor = self.context["trg"]  # -> (B, S', D)
        B = dec.shape[0]
        _dec = dec.reshape((B, -1))  # -> (B, *)
        _trg = trg.reshape((B, -1))  # -> (B, *)
        loss = self._loss_difference(dec, trg) + self._loss_difference(_dec, _trg)
        return loss

    def _loss_difference(self, y: torch.Tensor, t: torch.Tensor):
        assert y.shape == t.shape
        assert len(y.shape) > 1
        cos = self.cos(y, t)
        return self.mse(y, t) + self.mse(cos, torch.ones_like(cos))

    def calculate_scores(self, y: torch.Tensor, t: torch.Tensor) -> dict:
        import numpy
        from torchmetrics.functional import accuracy, bleu_score, rouge_score

        # accuracy
        acc = accuracy(
            y.argmax(dim=-1),
            t.argmax(dim=-1),
            task="multiclass",
            num_classes=self.n_out,
        )
        acc = acc.item()

        bleus = []
        rouges = {}
        for _y, _t in zip(y, t):
            # BLEU
            preds = " ".join(self.to_tokens(_y))
            labls = " ".join(self.to_tokens(_t))
            b = bleu_score(preds, [labls])
            bleus.append(b)

            # ROUGE
            # # cf. https://aclanthology.org/W04-1013.pdf
            pred_text = self.to_text(_y)
            labl_text = self.to_text(_t)
            r = rouge_score(pred_text, labl_text)
            for k, v in r.items():
                v = v.item()
                if k in rouges:
                    rouges[k].append(v)
                else:
                    rouges[k] = [v]

        # setup scores
        bleu = numpy.mean(bleus)
        scores = dict(accuracy=acc, bleu=bleu)

        for k, rouge_list in rouges.items():
            rouge = numpy.mean(rouge_list)
            scores[k] = rouge

        return scores
