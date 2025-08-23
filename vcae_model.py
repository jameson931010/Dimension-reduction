import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_TIME_DIM = 100
INPUT_CHANNEL_DIM = 128
class EMG128VCAE(nn.Module):
    model_type = "VCAE"
    FILTER_NUM = 8 # The number of convolution filter in intermediate layer
    K = 3                    # conv kernel
    P = 1                    # padding for K=3
    POOL_K = 2
    POOL_S = 2
    # bits: bits after quantization; input resolution bits = 16

    def __init__(self, num_pooling: int = 3, num_filter: int = 4, num_conv: int = 2):
        """
        num_pooling: the number of pooling layer in encoder (mirrored by unpool in decoder)
        num_filter: code depth (channels of final encoder conv)        
        num_conv: the number of convolution layer before each pooling
        """
        super().__init__()
        assert num_pooling >= 1
        assert num_filter >= 1
        
        # Encoder
        enc_blocks, pools, prepool_sizes, add_padding = [], [], [], []
        in_ch = 1 # For the first layer
        dim_t, dim_c = INPUT_TIME_DIM, INPUT_CHANNEL_DIM
        for _ in range(num_pooling):
            network = [
                nn.Conv2d(in_ch, self.FILTER_NUM, kernel_size=self.K, padding=self.P),
                nn.BatchNorm2d(self.FILTER_NUM),
                #nn.ReLU(inplace=True),
                nn.LeakyReLU(inplace=True),
            ]
            for _ in range(num_conv-1):
                network.extend([
                    nn.Conv2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.K, padding=self.P),
                    nn.BatchNorm2d(self.FILTER_NUM),
                    #nn.ReLU(inplace=True),
                    nn.LeakyReLU(inplace=True),
                ])
            enc_blocks.append(nn.Sequential(*network))
            #pools.append(nn.MaxPool2d(kernel_size=self.POOL_K, stride=self.POOL_S))
            pools.append(nn.Conv2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.POOL_K, stride=self.POOL_S))
            add_padding.append([dim_t & 1, dim_c & 1])
            prepool_sizes.append([dim_t, dim_c])
            dim_t, dim_c = dim_t // 2, dim_c // 2
            in_ch = self.FILTER_NUM # For remaining layer in encoder

        self.encoder_convs = nn.ModuleList(enc_blocks)
        self.encoder_pools = nn.ModuleList(pools)

        # Project to code
        self.to_mu = nn.Conv2d(self.FILTER_NUM, num_filter, kernel_size=self.K, padding=self.P)
        self.to_logvar = nn.Conv2d(self.FILTER_NUM, num_filter, kernel_size=self.K, padding=self.P)
        self.from_code = nn.Sequential(
            nn.ConvTranspose2d(num_filter, self.FILTER_NUM, kernel_size=self.K, padding=self.P, stride=1),
            nn.BatchNorm2d(self.FILTER_NUM),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.K, padding=self.P, stride=1),
            nn.BatchNorm2d(self.FILTER_NUM),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )

        # Decoder
        dec_blocks, unpools = [], []
        for i in range(num_pooling):
            unpools.append(nn.ConvTranspose2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.POOL_K, stride=self.POOL_S, output_padding=add_padding[i]))
            #unpools.append(nn.Upsample(size=prepool_sizes[i], mode='bilinear'))
            network = []
            for _ in range(num_conv):
                network.extend([
                    nn.ConvTranspose2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.K, padding=self.P),
                    nn.BatchNorm2d(self.FILTER_NUM),
                    #nn.ReLU(inplace=True),
                    nn.LeakyReLU(inplace=True),
                ])
            dec_blocks.append(nn.Sequential(*network))
        self.decoder_unpools = nn.ModuleList(unpools)
        self.decoder_blocks = nn.ModuleList(dec_blocks)

        # Final reconstruction
        self.reconstruct = nn.ConvTranspose2d(self.FILTER_NUM, 1, kernel_size=self.K, padding=self.P)

        # Buffers for exact unpool sizing

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        for conv, pool in zip(self.encoder_convs, self.encoder_pools):
            h = conv(h)
            h = pool(h)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        code = mu + torch.exp(0.5*logvar) * torch.randn_like(mu)
        return code, mu, logvar

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        h = self.from_code(h)
        for i in reversed(range(len(self.decoder_blocks))):
            h = self.decoder_unpools[i](h)
            h = self.decoder_blocks[i](h)

        h = self.reconstruct(h)
        return h


    def forward(self, x):
        h = x  # To preserve the input, as ReLU is done in place; [B,1,100,128]
        code, mu, logvar = self.encode(h)
        out = self.decode(code)
        return out, mu, logvar
