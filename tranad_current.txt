import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=500
    ):

        super().__init__()

        self.dropout = nn.Dropout(
            p=dropout
        )

        position = torch.arange(
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ) *
            (
                -math.log(10000.0)
                / d_model
            )
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


class TranAD(nn.Module):

    def __init__(
        self,
        n_features=9,
        window_size=60,
        hidden_size=32,
        n_heads=3,
        dropout=0.1
    ):

        super().__init__()

        self.name = "TranAD"

        self.n_features = n_features
        self.window_size = window_size

        self.input_projection = nn.Linear(
            n_features,
            2 * n_features
        )

        self.pos_encoder = PositionalEncoding(
            d_model=2 * n_features,
            dropout=dropout,
            max_len=window_size
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * n_features,
            nhead=n_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=False,
            activation="gelu"
        )

        self.transformer_encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=1
            )
        )

        decoder_layer_1 = (
            nn.TransformerDecoderLayer(
                d_model=2 * n_features,
                nhead=n_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=False,
                activation="gelu"
            )
        )

        decoder_layer_2 = (
            nn.TransformerDecoderLayer(
                d_model=2 * n_features,
                nhead=n_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=False,
                activation="gelu"
            )
        )

        self.transformer_decoder_1 = (
            nn.TransformerDecoder(
                decoder_layer_1,
                num_layers=1
            )
        )

        self.transformer_decoder_2 = (
            nn.TransformerDecoder(
                decoder_layer_2,
                num_layers=1
            )
        )

        self.output_projection = nn.Linear(
            2 * n_features,
            n_features
        )

    def encode(
        self,
        source,
        conditioning
    ):

        source = torch.cat(
            [
                source,
                conditioning
            ],
            dim=2
        )

        source = source * math.sqrt(
            self.n_features
        )

        source = self.pos_encoder(
            source
        )

        memory = self.transformer_encoder(
            source
        )

        return memory

    def decode(
        self,
        decoder,
        target,
        memory
    ):

        target = target.repeat(
            1,
            1,
            2
        )

        output = decoder(
            target,
            memory
        )

        output = self.output_projection(
            output
        )

        return output

    def forward(
        self,
        source
    ):

        source = source.permute(
            1,
            0,
            2
        )

        conditioning = torch.zeros_like(
            source
        )

        memory = self.encode(
            source,
            conditioning
        )

        reconstruction_1 = self.decode(
            self.transformer_decoder_1,
            source,
            memory
        )

        conditioning = (
            reconstruction_1 - source
        ) ** 2

        memory = self.encode(
            source,
            conditioning
        )

        reconstruction_2 = self.decode(
            self.transformer_decoder_2,
            source,
            memory
        )

        reconstruction_1 = (
            reconstruction_1.permute(
                1,
                0,
                2
            )
        )

        reconstruction_2 = (
            reconstruction_2.permute(
                1,
                0,
                2
            )
        )

        return (
            reconstruction_1,
            reconstruction_2
        )


def reconstruction_loss(
    source,
    reconstruction
):

    return torch.mean(
        torch.abs(
            source - reconstruction
        )
    )


def anomaly_score(
    source,
    reconstruction
):

    return torch.mean(
        (
            source - reconstruction
        ) ** 2,
        dim=(1, 2)
    )