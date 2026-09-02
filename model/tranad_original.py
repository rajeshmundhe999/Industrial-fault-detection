import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerDecoder
from torch.nn import TransformerDecoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=500):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32
            ) *
            (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            1,
            d_model
        )

        pe[:, 0, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 0, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        x = x + self.pe[
            :x.size(0)
        ]

        return self.dropout(x)


class TranADOriginal(nn.Module):

    def __init__(self, feats):

        super().__init__()

        self.n_feats = feats
        self.d_model = 2 * feats

        self.pos_encoder = PositionalEncoding(
            self.d_model,
            0.1
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=feats,
            dim_feedforward=16,
            dropout=0.1
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            1
        )

        decoder_layer_1 = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=feats,
            dim_feedforward=16,
            dropout=0.1
        )

        self.transformer_decoder1 = TransformerDecoder(
            decoder_layer_1,
            1
        )

        decoder_layer_2 = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=feats,
            dim_feedforward=16,
            dropout=0.1
        )

        self.transformer_decoder2 = TransformerDecoder(
            decoder_layer_2,
            1
        )

        self.fcn = nn.Sequential(
            nn.Linear(
                self.d_model,
                feats
            ),
            nn.Sigmoid()
        )

    def encode(
        self,
        src,
        c
    ):

        src = torch.cat(
            [
                src,
                c
            ],
            dim=2
        )

        src = src * math.sqrt(
            self.n_feats
        )

        src = self.pos_encoder(
            src
        )

        memory = self.transformer_encoder(
            src
        )

        return memory

    def forward(
        self,
        src,
        tgt
    ):

        c = torch.zeros_like(
            src
        )

        memory = self.encode(
            src,
            c
        )

        tgt = tgt.repeat(
            1,
            1,
            2
        )

        tgt = tgt * math.sqrt(
            self.n_feats
        )

        tgt = self.pos_encoder(
            tgt
        )

        x1 = self.transformer_decoder1(
            tgt,
            memory
        )

        x1 = self.fcn(
            x1
        )

        c = (
            x1 - src
        ) ** 2

        memory = self.encode(
            src,
            c
        )

        x2 = self.transformer_decoder2(
            tgt,
            memory
        )

        x2 = self.fcn(
            x2
        )

        return x1, x2