# 修改了 DSDCN
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import DeformConv2d
from einops import rearrange, reduce
import torchvision.utils as tvutils


# DWDCN
class DWConv2d(nn.Module):

    def __init__(
            self, in_ch, out_ch, kernel_size=1, stride=1, norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish, bn_weight_init=1, offset_clamp=(-3,3)
    ):
        super().__init__()

        self.offset_clamp=offset_clamp
        self.offset_generator=nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=3,
                                                      stride= 1,padding= 1,bias= False,groups=in_ch),
                                            nn.Conv2d(in_channels=in_ch, out_channels=18,
                                                      kernel_size=1, stride=1, padding=0, bias=False))
        self.dcn=DeformConv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,
                    stride= 1, padding= 1, bias= False, groups=in_ch)
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        offset = self.offset_generator(x)
        if self.offset_clamp:
            offset=torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])
        x = self.dcn(x, offset)
        x=self.pwconv(x)
        x = self.act(x)
        return x

class CDEMoudle(nn.Module):
    def __init__(self, in_ch, in_ch2, fourier_dim):
        super().__init__()
        
        self.layer1 = DWConv2d(in_ch, fourier_dim)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_ch2, fourier_dim, kernel_size=5, padding=2, dilation=1, padding_mode='reflect'),
            nn.Conv2d(fourier_dim, fourier_dim, kernel_size=1),
            nn.GELU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(fourier_dim * 2, fourier_dim * 2, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GELU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(fourier_dim * 2, fourier_dim * 2, kernel_size=1),
            nn.GELU()
        )

        self.conv3 = nn.Conv2d(fourier_dim * 4, fourier_dim, kernel_size=1)

    def forward(self, inp, dark):
        x1 = self.layer1(inp)
        x2 = self.layer2(dark)

        x = torch.concat([x1, x2], dim=1)
        x1 = self.layer3(x)
        x2 = self.layer4(x)

        x = self.conv3(torch.concat([x1, x2], dim = 1))

        return x


class DFEUnit(nn.Module):
    def __init__(self, dim):
        super(DFEUnit, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
    
        self.layer1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect'),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.conv1(x)
        
        out = self.layer1(x) * self.layer2(x)
        return out

class HFOUnit(nn.Module):
    def __init__(self, dim):
        super(HFOUnit, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True)
        )
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, inp, dark = None):
        x = self.conv1(inp)
        if dark is None:
            atten = self.sigmoid(self.ca(x) * self.pa(x))
        else:
            atten = self.sigmoid(self.ca(x) * self.pa(dark))
        out = x * atten
        return out

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class SimpleFusion(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 + x2

class DFFBlock(nn.Module):
    def __init__(self, dim, time_emb_dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.dfeunit = DFEUnit(dim)

        self.hfounit = HFOUnit(dim)

        self.mlp1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 1),
            nn.GELU(),
            # SimpleFusion(),
            nn.Conv2d(dim * 4, dim, 1)
        )

        self.mlp3 = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim * 6)
        ) if time_emb_dim else None


    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(6, dim=1)
    

    def forward(self, x):

        inp, time, dark = x

        shift_att, scale_att, shift_ffn, scale_ffn, shift_dark, scale_dark = self.time_forward(time, self.mlp3)
        
        # dark infor
        dark = dark * (scale_dark + 1) + shift_dark

        x = inp
        identity = x
        x = self.norm1(x)
        # MLP Time
        x = x * (scale_att + 1) + shift_att

        x = torch.cat([x, self.dfeunit(x * dark)], dim=1)
        x = self.mlp1(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        # MLP Time
        x = x * (scale_ffn + 1) + shift_ffn

        x = torch.cat([x, self.hfounit(x, dark)], dim=1)
        x = self.mlp2(x)
        x = identity + x

        return x

 
class BasicLayer(nn.Module):
    def __init__(self, dim, depth, time_emb_dim):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [DFFBlock(dim=dim, time_emb_dim = time_emb_dim) for i in range(depth)])

    def forward(self, x):
        inp, time, dark = x
        x = inp
        for blk in self.blocks:
            # print('000')
            x = blk([x, time, dark])
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class DADNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=3,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[1, 1, 2, 1, 1], refine_depth=2):
        super(DADNet, self).__init__()

        fourier_dim = 24
        time_dim = 24 * 4
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        self.cdemoudle = CDEMoudle(3, 1, fourier_dim)
 
        # setting
        self.patch_size = 4

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans * 2, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0], time_emb_dim=time_dim)

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)

        self.dark_down1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)


        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1], time_emb_dim=time_dim)

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)

        self.dark_down2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)
       

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2], time_emb_dim=time_dim)

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])
        
        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3], time_emb_dim=time_dim)

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        
        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4], time_emb_dim=time_dim)

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)
        

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, inp, cond, dark, time):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(inp.device)

        x = inp - cond
        inp_cond = x

        x = torch.cat([x, cond], dim=1)
        t = self.time_mlp(time)

        # dark_img = self.get_dark(cond)
        density0 = self.cdemoudle(inp, dark)
        density1 = self.dark_down1(density0)
        density2 = self.dark_down2(density1)

        x = self.patch_embed(x)
        x = self.layer1([x, t, density0])
        skip1 = x

        x = self.patch_merge1(x)

        x = self.layer2([x, t, density1])
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3([x, t, density2])
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4([x, t, density1])
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5([x, t, density0])
        x = self.patch_unembed(x)
        return x, inp_cond
    
    def get_dark(self, input):

        pool = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        values, _ = torch.min(input, dim=1, keepdim=True)
        
        input = pool(1 - values)

        return 1 - input

    def forward(self, inp, cond, time):
        H, W = inp.shape[2:]
        dark = self.get_dark(inp)
        inp = self.check_image_size(inp)
        cond = self.check_image_size(cond)
        dark = self.check_image_size(dark)

        feat, inp_cond = self.forward_features(inp, cond, dark, time)

        return inp + feat

def CondDADNet(opt):
    return DADNet(
        in_chans=opt['in_nc'],
        out_chans=opt['out_nc'],
        embed_dims=opt['embed_dims'],
        depths=opt['depths'],
        refine_depth=2)

if __name__ == '__main__':
    net = DADNet()
    # net = DWConv2d(3, 48)


    input = torch.randn((4, 3, 256, 256))
    cond = torch.randn((4, 3, 256, 256))

    output = net(input, cond, 100)
    print('output', output.shape)
