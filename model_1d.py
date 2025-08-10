import torch
import torch.nn as nn
import torch.nn.functional as F

class EMG128CAE(nn.Module):
    INPUT_TIME_DIM = 100
    INPUT_CHANNEL_DIM = 128
    #FILTER_NUM = 16 # The number of convolution filter in intermediate layer
    CON_KERNEL_SIZE = 3
    CON_PADDING = int((CON_KERNEL_SIZE-1)/2)
    POOL_KERNEL_SIZE = 2
    POOL_STRIDE = 2
    # Upsampling mode, try "nearest", "bilinear" or "bicubic". The latter is more time comsuming.
    POOL_MODE = "bilinear"
    # bits: bits after quantization; input resolution bits = 16

    def __init__(self, num_pooling=3, num_filter=2):
        """
        num_pooling: the number of pooling layer
        num_filter: the number of convolution filter in the last layer
        """
        super(EMG128CAE, self).__init__()
        
        # CR = (100*128) / (6*8) / 2 = 133.3
        encoder_layers = []

        # Encoder
        # First n-1 layer
        dim = self.INPUT_CHANNEL_DIM # The following assumes that dim is a power of 2
        for layer in range(num_pooling):
            encoder_layers.extend([
                nn.Conv1d(dim, dim, kernel_size=self.CON_KERNEL_SIZE, padding=self.CON_PADDING),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(),
                nn.Conv1d(dim, dim//2, kernel_size=self.POOL_KERNEL_SIZE, stride=self.POOL_STRIDE)
            ])
            dim //= 2

        """
        # Last convolution layer
        encoder_layers.extend([
            nn.Conv1d(dim, dim, kernel_size=self.CON_KERNEL_SIZE, padding=self.CON_PADDING),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
            nn.Conv1d(dim, num_filter, kernel_size=self.CON_KERNEL_SIZE, padding=self.CON_PADDING)
        ])
        """

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []

        # First layer
        power = 2 ** (num_pooling - 1)
        """
        additional_padding = int(self.INPUT_TIME_DIM//power) & 1 # whether downsampling have lose 1 dimension
        decoder_layers.extend([
            nn.Conv1d(num_filter, num_filter, kernel_size=self.CON_KERNEL_SIZE, padding=self.CON_PADDING),
            nn.BatchNorm1d(num_filter),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(num_filter, dim, kernel_size=self.POOL_KERNEL_SIZE, stride=self.POOL_STRIDE, output_padding=additional_padding)
        ])
        """

        # Other layers
        for layer in range(num_pooling):
            additional_padding = int(self.INPUT_TIME_DIM//power) & 1 # whether downsampling have lose 1 dimension
            decoder_layers.extend([
                nn.Conv1d(dim, dim, kernel_size=self.CON_KERNEL_SIZE, padding=self.CON_PADDING),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(),
                nn.ConvTranspose1d(dim, dim*2, kernel_size=self.POOL_KERNEL_SIZE, stride=self.POOL_STRIDE, output_padding=additional_padding)
            ])
            power //= 2
            dim *= 2

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1) # (Batchsize, 1, 100, 128) -> (Batchsize, 128, 100)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1).unsqueeze(1)
        return x
