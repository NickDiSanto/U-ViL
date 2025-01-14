import torch
from torch import nn

from model.encoder import build_uvil_encoder
from model.bottleneck import build_uvil_bottleneck
from model.decoder import build_uvil_decoder

class UViL(nn.Module):
    def __init__(self, encoder_config, bottleneck_config, decoder_config) -> None:
        super().__init__()
        self.encoder = build_uvil_encoder()
        self.bottleneck = build_uvil_bottleneck()
        self.decoder = build_uvil_decoder()

    def run_encoder(self, x):
        pass
    def run_bottleneck(self, x):
        pass
    def run_decoder(self, x):
        pass

    def forward(self, x):
        x = self.run_encoder(x)
        x = self.run_bottleneck(x)
        x = self.decoder(x)
        return x