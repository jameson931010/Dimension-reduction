import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_TIME_DIM = 100
INPUT_CHANNEL_DIM = 128
class EMG128CAE(nn.Module):
    model_type = "CAE"
    FILTER_NUM = 48 # The number of convolution filter in intermediate layer
    K = 3                    # conv kernel
    P = 1                    # padding for K=3
    POOL_K = 2 # Kernel for up-/down-sampling
    POOL_S = 2 # Stride for up-/down-sampling
    NUM_GROUP = 4 # The number of group for group normalization

    def __init__(self, num_pooling = 3, num_filter = 4, num_conv = 2):
        """
        num_pooling: The number of pooling layer in encoder (mirrored by unpool in decoder)
        num_filter: Code depth (channels of final encoder conv)        
        num_conv: The number of convolution layer before each pooling
        """
        super().__init__()
        
        # Encoder
        convs, pools, add_padding = [], [], []
        dim_t, dim_c = INPUT_TIME_DIM, INPUT_CHANNEL_DIM
        in_ch = 1 # For the first layer

        for i in range(num_pooling):
            network = [
                nn.Conv2d(in_ch, self.FILTER_NUM, kernel_size=self.K, padding=self.P),
                nn.GroupNorm(num_groups=self.NUM_GROUP, num_channels=self.FILTER_NUM),
                nn.LeakyReLU(inplace=True),
            ]
            for _ in range(num_conv-1):
                network.extend([
                    nn.Conv2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.K, padding=self.P),
                    nn.GroupNorm(num_groups=self.NUM_GROUP, num_channels=self.FILTER_NUM),
                    nn.LeakyReLU(inplace=True),
                ])
            convs.append(nn.Sequential(*network))
            pools.append(nn.Conv2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.POOL_K, stride=self.POOL_S))

            # For symmetric reconstruction
            add_padding.append([dim_t & 1, dim_c & 1])
            dim_t, dim_c = dim_t // 2, dim_c // 2

            # For remaining layer in encoder
            in_ch = self.FILTER_NUM 

        self.encoder_convs = nn.ModuleList(convs)
        self.encoder_pools = nn.ModuleList(pools)

        # Project to code
        proj = [
            nn.Conv2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.K, padding=self.P),
            nn.GroupNorm(num_groups=self.NUM_GROUP, num_channels=self.FILTER_NUM),
            nn.LeakyReLU(inplace=True),
        ] * 3
        proj.append(nn.Conv2d(self.FILTER_NUM, num_filter, kernel_size=self.K, padding=self.P))
        self.to_code = nn.Sequential(*proj)

        # Project from code
        self.from_code = nn.Sequential(
            nn.ConvTranspose2d(num_filter, self.FILTER_NUM, kernel_size=self.K, padding=self.P, stride=1),
            nn.GroupNorm(num_groups=self.NUM_GROUP, num_channels=self.FILTER_NUM),
            nn.LeakyReLU(inplace=True),
        )

        # Decoder
        convtrans, unpools = [], []
        for i in reversed(range(num_pooling)):
            unpools.append(nn.ConvTranspose2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.POOL_K, stride=self.POOL_S, output_padding=add_padding[i]))
            network = []
            for j in range(num_conv):
                network.extend([
                    nn.ConvTranspose2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.K, padding=self.P),
                    nn.GroupNorm(num_groups=self.NUM_GROUP, num_channels=self.FILTER_NUM),
                    nn.LeakyReLU(inplace=True),
                ])
            convtrans.append(nn.Sequential(*network))
        self.decoder_unpools = nn.ModuleList(unpools)
        self.decoder_convs = nn.ModuleList(convtrans)

        # Final reconstruction
        self.reconstruct = nn.Sequential(
            nn.ConvTranspose2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.K, padding=self.P),
            nn.GroupNorm(num_groups=self.NUM_GROUP, num_channels=self.FILTER_NUM),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(self.FILTER_NUM, 1, kernel_size=self.K, padding=self.P),
        )

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        for conv, pool in zip(self.encoder_convs, self.encoder_pools):
            h = conv(h)
            h = pool(h)
        h = self.to_code(h)
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        h = self.from_code(h)
        for unpool, conv in zip(self.decoder_unpools, self.decoder_convs):
            h = unpool(h)
            h = conv(h)
        h = self.reconstruct(h)
        return h

    def forward(self, x):
        h = x  # To preserve the input, as ReLU is done in place; [B,1,100,128]
        code = self.encode(h)
        out = self.decode(code)
        return out
