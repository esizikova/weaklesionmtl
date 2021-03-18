# adapted from https://github.com/sairin1202/Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/model.py

path = '../segmentation_models.pytorch/'

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


import sys
sys.path.append(path)
import segmentation_models_pytorch as smp
class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        # model = smp.Unet()
        #self.deeplab_model =  smp.DeepLabV3Plus(encoder_name='resnet101', encoder_weights='imagenet')
        self.deeplab_model =  smp.DeepLabV3Plus(encoder_name='resnet101')
        # encoder
        self.deeplab_encoder = self.deeplab_model.encoder
        #self.deeplab_encoder = self.deeplab_encoder.cuda(0)
        
        # self attention generation
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))#.cuda(0)
        
        self.mlp1a = nn.Linear(2048, 4096)#.cuda(0)
        self.mlp2a = nn.Linear(4096, 2048)#.cuda(0)
        
        self.mlp1b = nn.Linear(256, 4096)#.cuda(0)
        self.mlp2b = nn.Linear(4096, 256)#.cuda(0)
    
        self.upsample = nn.Upsample(16)#.cuda(0)

        # decoder
        self.deeplab_decoder = self.deeplab_model.decoder
        self.deeplab_decoder = self.deeplab_decoder#.cuda(1)
        self.deeplab_seghead = self.deeplab_model.segmentation_head
        # self.dec = Decoder(2, 512, 2, activ='relu', pad_type='reflect')

    def forward(self, x, y):
        # reconstruct an image
        vgg_x, vgg_y, vgg_x1_weight, vgg_x2_weight, vgg_y1_weight, vgg_y2_weight = self.encode(x, y)
        images_recon_x, images_recon_y = self.decode(
            vgg_x, vgg_y, vgg_x1_weight, vgg_x2_weight, vgg_y1_weight, vgg_y2_weight)
        return images_recon_x, images_recon_y

    def generate_attention(self, x):
        all_features_x = self.deeplab_encoder(x)
        
        features1_x = all_features_x[-1]
        features2_x = all_features_x[-4]

        feature1_x = self.global_avg_pool(features1_x)
#         attention1_x = self.upsample(F.sigmoid(self.mlp2a(
#             F.tanh(self.mlp1a(feature1_x.view(-1, 2048))))).view(-1, 2048, 1, 1))
        attention1_x = F.sigmoid(self.mlp2a(
            F.tanh(self.mlp1a(feature1_x.view(-1, 2048))))).view(-1, 2048, 1, 1)


        feature2_x = self.global_avg_pool(features2_x)
#         attention2_x = self.upsample(F.sigmoid(self.mlp2b(
#             F.tanh(self.mlp1b(feature2_x.view(-1, 256))))).view(-1, 256, 1, 1))
        attention2_x = F.sigmoid(self.mlp2b(F.tanh(self.mlp1b(feature2_x.view(-1, 256))))).view(-1, 256, 1, 1)


        return all_features_x, attention1_x, attention2_x


    def encode(self, x, y):
        # encode an image to its content
        vgg_x,attention1_x, attention2_x = self.generate_attention(x)
        vgg_y,attention1_y, attention2_y = self.generate_attention(y)

        return vgg_x, vgg_y, attention1_x, attention2_x, attention1_y, attention2_y


    def decode(self, vgg_x, vgg_y, attention1_x, attention2_x, attention1_y, attention2_y):

        attention1_x = nn.functional.interpolate(attention1_x,size=(vgg_x[-1].shape[2], vgg_x[-1].shape[3]))
        attention2_x = nn.functional.interpolate(attention2_x,size=(vgg_x[-4].shape[2], vgg_x[-4].shape[3]))

        attention1_y = nn.functional.interpolate(attention1_y,size=(vgg_y[-1].shape[2], vgg_y[-1].shape[3]))
        attention2_y = nn.functional.interpolate(attention2_y,size=(vgg_y[-4].shape[2], vgg_y[-4].shape[3]))

        vgg_x[-1] = vgg_x[-1] * attention1_x
        vgg_x[-4] = vgg_x[-4] * attention2_x
        
        vgg_y[-1] = vgg_y[-1] * attention1_y
        vgg_y[-4] = vgg_y[-4] * attention2_y



        # decode content to an image
        mask_x = self.deeplab_decoder(*vgg_x)
        mask_y = self.deeplab_decoder(*vgg_y)

        seg_x = self.deeplab_seghead(mask_x)
        seg_y = self.deeplab_seghead(mask_y)
        # mask_x = self.dec(attention_y * vgg_x)
        # mask_y = self.dec(attention_x * vgg_y)
        #print("image size:",images.size())
        return seg_x, seg_y 



class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # encoder
        self.pretrained_model = vgg16(pretrained=True)
        self.features, self.classifiers = list(self.pretrained_model.features.children(
        )), list(self.pretrained_model.classifier.children())
        self.features_map = nn.Sequential(*self.features)

        # self attention generation
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp1 = nn.Linear(512, 4096)
        self.mlp2 = nn.Linear(4096, 512)
        self.upsample = nn.Upsample(16)

        # decoder
        self.dec = Decoder(2, 512, 2, activ='relu', pad_type='reflect')

    def forward(self, x, y):
        # reconstruct an image
        vgg_x, vgg_y, vgg_x_weight, vgg_y_weight = self.encode(x, y)
        images_recon_x, images_recon_y = self.decode(
            vgg_x, vgg_y, vgg_x_weight, vgg_y_weight)
        return images_recon_x, images_recon_y

    def generate_attention(self, x):
        vgg_x = self.features_map(x)
        feature_x = self.global_avg_pool(vgg_x)
        attention_x = self.upsample(F.sigmoid(self.mlp2(
            F.tanh(self.mlp1(feature_x.view(-1, 512))))).view(-1, 512, 1, 1))
        return vgg_x,attention_x

    def encode(self, x, y):
        # encode an image to its content
        vgg_x,attention_x = self.generate_attention(x)
        vgg_y,attention_y = self.generate_attention(y)

        return vgg_x, vgg_y, attention_x, attention_y


    def decode(self, vgg_x, vgg_y, attention_x, attention_y):
        # decode content to an image
        mask_x = self.dec(attention_y * vgg_x)
        mask_y = self.dec(attention_x * vgg_y)
        #print("image size:",images.size())
        return mask_x, mask_y


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='bn', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm,
                                    activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_res, dim, output_dim, activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, 'bn', activ, pad_type=pad_type)]

        for i in range(5):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='bn', activation=activ, pad_type='reflect')]
            dim //= 2

        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3,
                                   norm='none', activation='none', pad_type='reflect')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='bn', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm,
                              activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm,
                              activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)

        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=-1)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim,
                              kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
