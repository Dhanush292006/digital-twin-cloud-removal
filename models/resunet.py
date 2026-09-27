from .unet import UNet


class ResUNet(UNet):
    """Residual-output U-Net baseline."""
    def __init__(self, in_channels, out_channels, base=32):
        super().__init__(in_channels, out_channels, base, residual=True)
