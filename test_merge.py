import random

import torch

# from models.rfdn_half.RFDNB2 import RFDNB2 as model, RFDNB2_P as model_p
from models.rfdn_half.RFDNB2S import RFDNB2S as model, RFDNB2S_P as model_p
# from models.rfdn_half.RFDNB4 import RFDNB4 as model, RFDNB4_P as model_p


def main():
    in_channels = random.randint(4, 10)
    convs = model(in_nc=in_channels)
    conv = model_p(convs, in_nc=in_channels)
    batch = random.randint(4, 10)
    inp = torch.randn(batch, in_channels, 224, 224)
    conv_output = conv(inp)
    print(conv_output.shape)
    convs_output = convs(inp)
    print(convs_output.shape)
    print(conv_output - convs_output)
    print(torch.sum(torch.abs(conv_output - convs_output)))


if __name__ == "__main__":
    main()
