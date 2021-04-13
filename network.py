import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

import math

import torch
from torch import nn
from torch.nn import functional as F

from stylegan2.model import StyledConv, Blur, EqualLinear, EqualConv2d, ScaledLeakyReLU
from stylegan2.op import FusedLeakyReLU



class Discriminator_Y(nn.Module):
    def __init__(self, size, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        dims = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64
#             512: 32,
#             1024: 16,
        }
        
        log_size = int(math.log(size, 2))
        in_dim = dims[size]
        
        blocks = []
        for i in range(log_size, 2, -1):
            out_dim = dims[2 ** (i - 1)]
            blocks.append(ResBlock(in_dim, out_dim, blur_kernel))
            in_dim = out_dim
                          
        self.blocks = nn.Sequential(*blocks)

        self.final_conv = ConvLayer(in_dim, dims[4], 3)

        self.final_linear = nn.Sequential(
            EqualLinear(dims[4] * 8 * 8, dims[4], activation="fused_lrelu"),
            EqualLinear(dims[4], 1),
        )
                          

    def forward(self, input):
        out = self.blocks(input)
#         print('0', out.size())
        out = self.final_conv(out)
#         print('1',out.size())
        out = out.view(out.shape[0], -1)
#         print('2', out.size())
        out = self.final_linear(out)
        return out
        
                          
class Discriminator_X(nn.Module):
    def __init__(self, size, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        dims = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64
#             512: 32,
#             1024: 16,
        }
        
        log_size = int(math.log(size, 2))
        in_dim = dims[size]
        
        blocks = []
        for i in range(log_size, 2, -1):
            out_dim = dims[2 ** (i - 1)]
            blocks.append(ResBlock(in_dim, out_dim, blur_kernel))
            in_dim = out_dim
                          
        self.blocks = nn.Sequential(*blocks)

        self.final_conv = ConvLayer(in_dim, dims[4], 3)

        self.final_linear = nn.Sequential(
            EqualLinear(dims[4] * 8 * 8, dims[4], activation="fused_lrelu"),
            EqualLinear(dims[4], 1),
        )
                    

    def forward(self, input):
#         print('input', input.size())
        out = self.blocks(input)
#         print('0', out.size())
        out = self.final_conv(out)
#         print('1',out.size())
        out = out.view(out.shape[0], -1)
#         print('2', out.size())
        out = self.final_linear(out)
        return out    
                      
                          
class Discriminator_U(nn.Module):
    def __init__(self, size, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        
        dims = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64
#             512: 32,
#             1024: 16,
        }

        in_dim = 3
        blocks = [ConvLayer(3, dims[size], 1)]
        log_size = int(math.log(size, 2))
        in_dim = dims[size]

        for i in range(log_size, 5, -1):
            out_dim = dims[2 ** (i - 1)]
            blocks.append(ResBlock(in_dim, out_dim, blur_kernel))
            in_dim = out_dim

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        out = self.blocks(input)
        return out

    
    

class EqualConvTranspose2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


    
class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=(1, 3, 3, 1),
        bias=True,
        activate=True,
        stride=1, 
        padding="zero",
    ):
        layers = []

        self.padding = 0
#         stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2

        if upsample:
            layers.append(
                EqualConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

        else:
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2

                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2

                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))

                    self.padding = 0

                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk):
        super(MLP, self).__init__()
        self.model = []
        self.model = [EqualLinear(input_dim, dim, activation="fused_lrelu")]
        for i in range(n_blk - 2):
            self.model += [EqualLinear(dim, dim, activation="fused_lrelu")]
        self.model += [EqualLinear(dim, output_dim)] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
#         print(x.size())
#         print(x.view(x.size(0), -1).size())
        return self.model(x.view(x.size(0), -1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True, padding="zero"):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, out_channel, 3, padding=padding)
        self.conv2 = ConvLayer(out_channel, out_channel, 3, downsample=downsample, padding=padding)

        if downsample or in_channel != out_channel:
            self.skip = ConvLayer(
                in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
            )
        else:
            self.skip = None

    def forward(self, input):
#         print('res input', input.size())
        out = self.conv1(input)
#         print('b', out.size())
        out = self.conv2(out)
        # print('res output1', out.size())
        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input

        # print('out', out.size())
        # print('skip', skip.size())
        out = (out + skip) / math.sqrt(2)
        # print('res output2', skip.size())
        return out
                          
                          
    
# class ResBlock(nn.Module):
#     def __init__(self, in_dim, out_dim, norm='in', activation='relu', pad_type='zero', downsample=False):
#         super(ResBlock, self).__init__()

#         self.downsample = downsample

#         model = []
#         model += [Conv2dBlock(in_dim, out_dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
#         model += [Conv2dBlock(out_dim, out_dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]

#         if self.downsample:
#             self.residual = Conv2dBlock(in_dim, out_dim, 4, 2, 1, norm=norm, activation=activation, pad_type=pad_type)
#             model += [nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)]

#         self.model = nn.Sequential(*model)

#     def forward(self, x):
# #         print('res input', x.size())
#         if self.downsample:
#             residual = self.residual(x)
#         else:
#             residual = x 
# #         print('skip ouput', residual.size())
#         out = self.model(x)
# #         print('res ouput', out.size())

# #         print('out', out.size())
# #         print('residual', residual.size())
#         out += residual
#         return out

    
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, in_dim, out_dim, downsample=True):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(in_dim, out_dim, downsample)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
    
                          

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.conv1 = ConvLayer(3, self.in_channels, 1, stride=1)                                                             
        self.blocks = []
        
        for i in range(4):
            self.blocks.append(ResBlock(self.in_channels, self.in_channels*2, padding="reflect"))
            self.in_channels = self.in_channels * 2
        self.blocks.append(ResBlock(self.in_channels, self.in_channels, padding="reflect"))   # 추가 
        self.blocks = nn.Sequential(*self.blocks)

        self.structure = nn.Sequential(ConvLayer(self.in_channels, self.in_channels, 1, padding="valid"),
                                       ConvLayer(512, 512, 1, stride=1, padding="valid")   # 512 channels
                                       )

        self.texture = nn.Sequential(ConvLayer(self.in_channels, self.in_channels * 2, 3, stride=2, padding="valid"),
                                     ConvLayer(self.in_channels * 2, self.in_channels * 4, 3, stride=2, padding="valid"),
                                     nn.AdaptiveAvgPool2d(1),                                                              # 
                                     ConvLayer(2048, 2 * 64 * 9, 1, stride=1, padding="valid")
                                     )

    def forward(self, input):
        # print('encoder start')
        out = self.conv1(input)
        out = self.blocks(out)
        # print('fn_out', out.size())
        s_code = self.structure(out)
#         print('s_code', s_code.size())
        t_code = self.texture(out)
#         t_code = torch.flatten(t_code, 1)
#         print('t_code', t_code.size())
#         print('s_code', s_code.size())
        # print('encoder finish')
        return s_code, t_code

                          
class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [ConvLayer(input_dim, dim, 3)]
#         self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [ConvLayer(dim, dim * 2, 3, downsample=True)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, dim, downsample=False)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)



class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [ConvLayer(input_dim, dim, 3)]                         
        for i in range(2):
            self.model += [ConvLayer(dim, dim * 2, 3)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [ConvLayer(dim, dim * 2, 3, downsample=True)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim * 2, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
#         self.output_dim = dim

        self.adain_param_num = 2 * style_dim * 7   # (gamma,beta;2) * (IL+LN;2) * (dims)  2 x 32 = 64
#         self.adain_param_num = 2 * style_dim * 2  # (gamma,beta;2) * (IL+LN;2) * (dims)  2 x 2 x 8 = 32
#         self.mlp = MLP(style_dim, 512*2, 256 , 3, norm='none', activ=activ)
        self.mlp = MLP(style_dim, self.adain_param_num, 256 , 3)
#         self.mlp = LinearBlock(style_dim, self.adain_param_num, norm='none', activation='none')

    def forward(self, x):
        style_code = self.model(x)
#         print('style_code', style_code.shape)
        adain_params = self.mlp(style_code)                                                                             # 각 이미지 마다 feature maps x 2
#         adain_params = torch.squeeze(torch.squeeze(ada_params,-1),-1)
#         print(adain_params.size())
        return adain_params


class PoLIN(nn.Module):
    def __init__(self, dim):
        super(PoLIN, self).__init__()
        self.conv1x1 = nn.Conv2d(dim*2, dim, 1, 1, 0, bias=False)

    def forward(self, input):
#         IN = nn.InstanceNorm2d(input.size()[1], affine=False)(input)
        LN = nn.LayerNorm(input.size()[1:], elementwise_affine=False)(input)
        IN = nn.InstanceNorm2d(input.size()[1], affine=False)(input)
        LIN = torch.cat((IN,LN),dim=1)
        result = self.conv1x1(LIN)
        return result

class Ada_PoLIN(nn.Module):
    def __init__(self, dim):
        super(Ada_PoLIN, self).__init__()
        self.Conv1x1 = nn.Conv2d(dim*2, dim, 1, 1, 0, bias=False)

    def forward(self, input, params):
#         print('input', input.size())
        IN = nn.InstanceNorm2d(input.size()[1], affine=False)(input)
#         LN = nn.InstanceNorm2d(input.size()[1], affine=False)(input)
#         IN = nn.LayerNorm(input.size()[1:], elementwise_affine=False)(input)
        LN = nn.LayerNorm(input.size()[1:], elementwise_affine=False)(input)
        LIN = torch.cat((IN,LN),dim=1)
        b,c,w,h = LIN.size()
#         print(b,c,w,h)
#         print('LIN size', LIN.size())
#         print('params_zie', params.size())
        mid = params.size()[1] // 2
#         print(params.size())
        gamma = params[:, :mid]
        beta = params[:, mid:]
        c = self.Conv1x1(LIN)
#         print('c', c.size())
#         print('pass')
#         print(c.size())
#         print(gamma.size())
#         print(beta.size())
#         result = gamma[0] * c + beta    
#         print('gamma', gamma.size())
#         print('beta', beta.size())
#         gamma= gamma.repeat(1,w*w).reshape((b,mid,w,w))
#         beta= beta.repeat(1,w*w).reshape((b,mid,w,w))
#         print('gamma', gamma.size())
#         print('beta', beta.size())
#         print('pass 2')
        gamma= gamma.unsqueeze(2).unsqueeze(3)
        beta= beta.unsqueeze(2).unsqueeze(3)
        result = gamma * c + beta
#         print(result.shape)
#         print('pass 3')
#         result = gamma.expand(-1,-1,w,h) + c + beta.expand(-1,-1,w,h)                                                                           # (16 x 1024) * (16 x 1024 x 8 x 8) : bcast 테스트필요
        return result



class ASC_block(nn.Module):
    def __init__(self, input_dim, dim, num_ASC_layers):
        super(ASC_block, self).__init__()
        self.input_dim = input_dim
        self.num_ASC_layers = num_ASC_layers
        self.ConvLayer = []
        self.NormLayer = []
        self.Ada_PoLINLayer = []
        for _ in range(self.num_ASC_layers):
            self.ConvLayer += [ConvLayer(self.input_dim, dim, 3)]
            self.NormLayer += [Ada_PoLIN(dim)]
            self.input_dim = dim
        self.ConvLayer = nn.ModuleList(self.ConvLayer)
        self.NormLayer = nn.ModuleList(self.NormLayer)

    def forward(self, x, Ada_PoLIN_params): # param은 encoder로 부터 온다.
#         print('Ada_Polin', Ada_PoLIN_params.size())
#         print('x_start', x.size())
#         print('bbbbbbb')
        idx = 0
        for ConvLayer, NormLayer in zip(self.ConvLayer, self.NormLayer):
#             print('loop')
#             print('x_size', x.size())
            x = ConvLayer(x)
#             print('x_size', x.size())
#             print('ada',Ada_PoLIN_params.size() )
#             print('ada__',Ada_PoLIN_params[..., idx].size() )
            x = NormLayer(x,Ada_PoLIN_params[..., idx])               
            idx += 1 
#             print('x_end', x.size())# Ada_PoLIN_params: feature maps x 2
        return x


class FST_block(nn.Module):
    def __init__(self, input_dim, dim):
        super(FST_block, self).__init__()
        self.input_dim = input_dim
        self.block = []
        self.block += [ConvLayer(self.input_dim, dim, 1, upsample=True, bias=False, activate=False)]       
        self.block += [PoLIN(dim)]
        self.block += [ConvLayer(dim, dim, 3)]                                    
        self.block = nn.Sequential(*self.block)
        self.Ada_PoLIN = Ada_PoLIN(dim)                                                                     
                      
    def forward(self, x, Ada_PoLIN_params): # param은 encoder로 부터 온다.
        x = self.block(x)
#         print('heree')
        x = self.Ada_PoLIN(x, Ada_PoLIN_params)
#         print('aaa')
        return x



# From MUNIT
class Decoder(nn.Module):
    def __init__(self, dim, output_dim, input_dim=512, num_ASC_layers=4, num_FST_blocks=5):
        super(Decoder, self).__init__()
        self.ASC_block = []
        self.FST_block = []
        
        # ASC Blocks
        self.ASC_block = ASC_block(input_dim, dim, num_ASC_layers)

        # FST Blocks
        for i in range(num_FST_blocks):
            self.FST_block += [FST_block(dim, dim)]
        self.FST_block = nn.ModuleList(self.FST_block)

        # Last Convlayer
        self.conv = ConvLayer(dim, 3, 1, activate=False)

    def forward(self, x, Ada_PoLIN_params):
#         print('aaaaaaaaaaaa')
#         print('x_size 0', x.size())
#         print('Ada_PoLIN_params', Ada_PoLIN_params.size())
        Ada_PoLIN_params = Ada_PoLIN_params.reshape((-1, 128, 9))
#         print( Ada_PoLIN_params[..., :4].size())
#         x = self.ASC_block(x, Ada_PoLIN_params=Ada_PoLIN_params)
        x = self.ASC_block(x, Ada_PoLIN_params[..., :4])
#         print('x_size 1', x.size())

        for idx, block in enumerate(self.FST_block):                
            x = block(x, Ada_PoLIN_params[..., idx+4])        
#             print('x_size 2', x.size())

        x = self.conv(x)
        return x



# class Decoder(nn.Module):
#     def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
#         super(Decoder, self).__init__()

#         self.model = []
#         # AdaIN residual blocks
#         self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
#         # upsampling blocks
#         for i in range(n_upsample):
#             self.model += [nn.Upsample(scale_factor=2),
#                            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
#             dim //= 2
#         # use reflection padding in the last conv layer
#         self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
#         self.model = nn.Sequential(*self.model)

#     def forward(self, x):
#         return self.model(x)




