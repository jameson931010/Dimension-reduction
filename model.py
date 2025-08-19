import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_TIME_DIM = 100
INPUT_CHANNEL_DIM = 128
class EMG128CAE(nn.Module):
    model_type = "CAE"
    FILTER_NUM = 64 # The number of convolution filter in intermediate layer
    K = 3                    # conv kernel
    P = 1                    # padding for K=3
    POOL_K = 2
    POOL_S = 2
    # bits: bits after quantization; input resolution bits = 16

    def __init__(self, num_pooling: int = 3, num_filter: int = 4, num_conv: int = 1):
        """
        num_pooling: the number of pooling layer in encoder (mirrored by unpool in decoder)
        num_filter: code depth (channels of final encoder conv)        
        num_conv: the number of convolution layer before each pooling
        """
        super().__init__()
        assert num_pooling >= 1
        assert num_filter >= 1
        
        # Encoder
        enc_blocks, pools = [], []
        in_ch = 1 # For the first layer
        for _ in range(num_pooling):
            network = [
                nn.Conv2d(in_ch, self.FILTER_NUM, kernel_size=self.K, padding=self.P),
                #nn.BatchNorm2d(self.FILTER_NUM),
                nn.ReLU(inplace=True),
                #nn.LeakyReLU(inplace=True),
            ]
            for _ in range(num_conv-1):
                network.extend([
                    nn.Conv2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.K, padding=self.P),
                    #nn.BatchNorm2d(self.FILTER_NUM),
                    nn.ReLU(inplace=True),
                    #nn.LeakyReLU(inplace=True),
                ])
            enc_blocks.append(nn.Sequential(*network))
            pools.append(nn.MaxPool2d(kernel_size=self.POOL_K, stride=self.POOL_S, return_indices=True))
            in_ch = self.FILTER_NUM # For remaining layer in encoder

        self.encoder_convs = nn.ModuleList(enc_blocks)
        self.encoder_pools = nn.ModuleList(pools)

        # Project to code
        self.to_code = nn.Conv2d(self.FILTER_NUM, num_filter, kernel_size=self.K, padding=self.P)
        self.from_code = nn.Sequential(
            nn.ConvTranspose2d(num_filter, self.FILTER_NUM, kernel_size=self.K, padding=self.P, stride=1),
            #nn.BatchNorm2d(self.FILTER_NUM),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(inplace=True),
        )

        # Decoder
        dec_blocks, unpools = [], []
        for _ in range(num_pooling):
            unpools.append(nn.MaxUnpool2d(kernel_size=self.POOL_K, stride=self.POOL_S))
            network = []
            for _ in range(num_conv):
                network.extend(nn.Sequential(
                    nn.ConvTranspose2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.K, padding=self.P),
                    #nn.BatchNorm2d(self.FILTER_NUM),
                    nn.ReLU(inplace=True),
                    #nn.LeakyReLU(inplace=True),
                ))
            dec_blocks.append(nn.Sequential(*network))
        self.decoder_unpools = nn.ModuleList(unpools)
        self.decoder_blocks = nn.ModuleList(dec_blocks)

        # Final reconstruction
        self.reconstruct = nn.ConvTranspose2d(self.FILTER_NUM, 1, kernel_size=self.K, padding=self.P)

        # Buffers for exact unpool sizing
        self._pool_indices = []
        self._prepool_sizes = []

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        for conv, pool in zip(self.encoder_convs, self.encoder_pools):
            h = conv(h)
            self._prepool_sizes.append(h.size())
            h, idx = pool(h)
            self._pool_indices.append(idx)
        h = self.to_code(h)
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        h = self.from_code(h)
        for i in reversed(range(len(self.decoder_blocks))):
            h = self.decoder_unpools[i](h, self._pool_indices[i], output_size=self._prepool_sizes[i])
            h = self.decoder_blocks[i](h)

        h = self.reconstruct(h)
        return h


    def forward(self, x):
        self._pool_indices.clear()
        self._prepool_sizes.clear()

        h = x  # To preserve the input, as ReLU is done in place; [B,1,100,128]
        code = self.encode(h)
        out = self.decode(code)
        return out
