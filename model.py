import torch
import torch.nn as nn
import torch.nn.functional as F

class EMG128CAE(nn.Module):
    INPUT_TIME_DIM = 100
    INPUT_CHANNEL_DIM = 128
    FILTER_NUM = 64 # The number of convolution filter in intermediate layer
    CON_KERNEL_SIZE = 3
    CON_PADDING = int((CON_KERNEL_SIZE-1)/2)
    POOL_KERNEL_SIZE = 2
    POOL_STRIDE = 2
    # Upsampling mode, try "linear", "bilinear" or "bicubic". The latter is more time comsuming.
    POOL_MODE = "bilinear"
    # bits: bits after quantization; input resolution bits = 16

    def __init__(self, num_pooling=4, num_filter=2):
        """
        num_pooling: the number of pooling layer
        num_filter: the number of convolution filter in the last layer
        """
        super(EMG128CAE, self).__init__()
        
        # CR = (100*128) / (6*8) / 2 = 133.3
        self.encoder = nn.Sequential(
            # First layer
            nn.Conv2d(1, self.FILTER_NUM, kernel_size=self.CON_KERNEL_SIZE, padding=self.CON_PADDING), # Zero_padded by default
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.POOL_KERNEL_SIZE, stride=self.POOL_STRIDE),

            # Intermediate layer
            *([
                nn.Conv2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.CON_KERNEL_SIZE, padding=self.CON_PADDING),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=self.POOL_KERNEL_SIZE, stride=self.POOL_STRIDE)
            ] * (num_pooling - 1)),

            # Last convolution layer
            nn.Conv2d(self.FILTER_NUM, num_filter, kernel_size=self.CON_KERNEL_SIZE, padding=self.CON_PADDING),
        )

        decoder_layers = []

        # First layer
        decoder_layers.extend([
            nn.ConvTranspose2d(num_filter, self.FILTER_NUM, kernel_size=self.CON_KERNEL_SIZE, padding=self.CON_PADDING),
            nn.ReLU()
        ])

        # Intermediate layer
        power = 2 ** (num_pooling - 1)
        for layer in range(num_pooling-1):
            decoder_layers.extend([
                nn.Upsample(size=(int(self.INPUT_TIME_DIM//power), int(self.INPUT_CHANNEL_DIM//power)), mode=self.POOL_MODE),
                nn.ConvTranspose2d(self.FILTER_NUM, self.FILTER_NUM, kernel_size=self.CON_KERNEL_SIZE, padding=self.CON_PADDING),
                nn.ReLU()
            ])
            power *= 2

        # Final layer
        decoder_layers.extend([
            nn.Upsample(size=(self.INPUT_TIME_DIM, self.INPUT_CHANNEL_DIM), mode=self.POOL_MODE),
            nn.ConvTranspose2d(self.FILTER_NUM, 1, kernel_size=self.CON_KERNEL_SIZE, padding=self.CON_PADDING)
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
