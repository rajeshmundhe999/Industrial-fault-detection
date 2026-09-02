import torch
import torch.nn as nn


class GRUAutoencoder(nn.Module):

    def __init__(self, input_size=9):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.GRU(
                input_size=input_size,
                hidden_size=32,
                batch_first=True
            )
        )

        self.gru2 = nn.GRU(
            input_size=32,
            hidden_size=16,
            batch_first=True
        )

        self.gru3 = nn.GRU(
            input_size=16,
            hidden_size=8,
            batch_first=True
        )

        self.decoder1 = nn.GRU(
            input_size=8,
            hidden_size=16,
            batch_first=True
        )

        self.decoder2 = nn.GRU(
            input_size=16,
            hidden_size=32,
            batch_first=True
        )

        self.output = nn.Linear(
            32,
            input_size
        )

    def forward(self, x):

        x, _ = self.encoder[0](x)

        x, _ = self.gru2(x)

        x, _ = self.gru3(x)

        x, _ = self.decoder1(x)

        x, _ = self.decoder2(x)

        x = self.output(x)

        return x