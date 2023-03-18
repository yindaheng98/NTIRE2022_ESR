# import sys
# sys.path.append('/home/data/NTIRE2022_ESR')
import torch
import torch.nn as nn
import models.rfdn_baseline.block as B
from models.rfdn_baseline.block import conv_layer, activation, ESA
from .RFDN import RFDN
from .block import conv_layer_expand,conv_layer,conv_block_exp,pixelshuffle_block_exp,ESA_EXP
import cv2
import numpy as np
exp = [2,2]

class RFDNEXP(RFDN):
    def __init__(self, model: RFDN,in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4,**kwargs):
        super().__init__()
        self.fea_conv = conv_layer_expand(model.fea_conv,in_nc,nf,[1,2],3)
        
        self.B1 = RFDB_EXP(in_channels=nf,model = model.B1)
        self.B2 = RFDB_EXP(in_channels=nf,model = model.B2)
        self.B3 = RFDB_EXP(in_channels=nf,model = model.B3)
        self.B4 = RFDB_EXP(in_channels=nf,model = model.B4)
        
        self.c = conv_block_exp(nf * num_modules, nf, kernel_size=1, act_type='lrelu',model = model.c)
        
        self.LR_conv = conv_layer_expand(model.LR_conv,nf,nf,exp,3)
        
        upsample_block = pixelshuffle_block_exp
        
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale,model=model.upsampler)
        
        self.scale_idx = 0

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
    

class RFDB_EXP(nn.Module):
    def __init__(self,model ,in_channels, distillation_rate=0.25):
        super(RFDB_EXP, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer_expand(model.c1_d,in_channels,self.dc,exp,1)
        self.c1_r = conv_layer_expand(model.c1_r,in_channels,self.rc,exp,3)
        self.c2_d = conv_layer_expand(model.c2_d,self.remaining_channels,self.dc,exp,1)
        self.c2_r = conv_layer_expand(model.c2_r,self.remaining_channels,self.rc, exp, 3)
        self.c3_d = conv_layer_expand(model.c3_d,self.remaining_channels, self.dc,exp, 1)
        self.c3_r = conv_layer_expand(model.c3_r,self.remaining_channels, self.rc,exp, 3)
        self.c4 = conv_layer_expand(model.c4,self.remaining_channels,self.dc,exp,3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer_expand(model.c5,self.dc*4,in_channels,exp,1)
        self.esa = ESA_EXP(in_channels, model.esa)


    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out)) 

        return out_fused

# test 
# in_c = 50
# out_c = 60
# exp = [1.5,1.5]
# batch = 4
# sample_input = torch.randn(batch, in_c, 123, 456)
# sample_input_big = torch.randn(batch, math.ceil(in_c*exp[0]), 123, 456)
# sample_input_big[0:batch, 0:in_c, ...] = sample_input
# # conv,in_channels, out_channels,expand,kernel_size
# conv = nn.Conv2d(in_c, out_c, 3)
# conv_big = conv_layer_expand(conv,in_c,out_c,exp,3)

# output = conv(sample_input)
# output_big = conv_big(sample_input_big)
# print(output)
# print(output.shape)
# print(output_big.shape)
# print(torch.mean(torch.abs(output_big[:, 0:out_c, ...] - output) / torch.abs(output)))
# a = RFDN()
if __name__ == '__main__':
    b =RFDN()
    a = RFDNEXP(b)
    
    img = cv2.imread('/home/data/NTIRE2022_ESR/a.png', 1).astype(np.uint8)
    img = img / 255.
    img = torch.from_numpy(img)
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    # out = b.forward(img)
    out = a.forward(img)



