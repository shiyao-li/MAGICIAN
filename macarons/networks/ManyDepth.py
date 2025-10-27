import os
from torchvision import models, transforms
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision
import math
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_apply
# from ..model_utils import *

dir_path = os.path.abspath(os.path.dirname(__file__))
weights_path = os.path.join(dir_path, "../../weights/resnet")

# -----Model parameters-----
input_height = 256
input_width = 456
input_channels = 3

d_min = 0.5  # Raw: 0.5 ; Normalized: 0.1
d_max = 750  # Raw: 750 ; Normalized: 418
n_alpha = 2
n_depth = 96

pose_factor = 100.
learn_pose = False


# -----Classes-----
class FeatureExtractor(nn.Module):
    def __init__(self, resnet_model):
        super(FeatureExtractor, self).__init__()

        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer = resnet_model.layer1

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.maxpool(res)
        res = self.layer(res)

        return res


def save_feature_extractor_from_resnet18(save_path, device):
    # Load pretrained ResNet18 model
    resnet_model = torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=True).to(device)
    resnet_model.eval()

    # Creates a feature extractor from the first layer
    feature_extractor = FeatureExtractor(resnet_model).to(device)

    # Save the whole model
    torch.save(feature_extractor, save_path)


def load_feature_extractor(path, device):
    feature_extractor = torch.load(path, map_location=device)
    return feature_extractor


def transpose_channels(img, channel_is_at_the_end):
    if channel_is_at_the_end:
        res = 0. + torch.transpose(img, dim0=-1, dim1=-2)
        res = torch.transpose(res, dim0=-2, dim1=-3)
    else:
        res = 0. + torch.transpose(img, dim0=-3, dim1=-2)
        res = torch.transpose(res, dim0=-2, dim1=-1)
    return res


class CostVolumeBuilder(nn.Module):
    def __init__(self, height, width,
                 feature_height, feature_width, feature_channels, n_alpha,
                 d_min, d_max, n_depth, output_channels,
                 kernel_size=3, stride=1, padding=1):
        super(CostVolumeBuilder, self).__init__()

        self.height = height
        self.width = width

        self.feature_height = feature_height
        self.feature_width = feature_width
        self.feature_channels = feature_channels

        self.n_alpha = n_alpha

        self.x_tab = torch.Tensor([[i for j in range(width)] for i in range(height)])
        self.y_tab = torch.Tensor([[j for j in range(width)] for i in range(height)])

        self.d_min = d_min
        self.d_max = d_max
        self.n_depth = n_depth
        self.depth_bins = torch.linspace(d_min, d_max, n_depth)

        self.conv_reduce = torch.nn.Conv2d(in_channels=feature_channels + n_depth,
                                           out_channels=output_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding)
        self.relu = torch.nn.ReLU()

    def reproject_depth_map(self, depth, cameras, features=None):
        """

        :param depth: Tensor with shape (n_images, height, width, 1)
        :param cameras: cameras.R should be a Tensor with shape (n_images, 3, 3),
        cameras.T should be a Tensor with shape (n_images, 3)
        and cameras.zfar should be a Tensor with shape (n_images, )
        :param features: Tensor with shape (n_images, height, width, channels)
        :return:
        """
        batch_size = depth.shape[0]
        height = depth.shape[-3]
        width = depth.shape[-2]

        if height != self.height or width != self.width:
            raise NameError("Depth has wrong dimensions.")

        ndc_x_tab = width / min(width, height) - (self.y_tab / (min(width, height) - 1)) * 2
        ndc_y_tab = height / min(width, height) - (self.x_tab / (min(width, height) - 1)) * 2

        ndc_points = torch.cat((ndc_x_tab.view(1, -1, 1).expand(batch_size, -1, -1),
                                ndc_y_tab.view(1, -1, 1).expand(batch_size, -1, -1),
                                depth.view(batch_size, -1, 1)),
                               dim=-1).view(batch_size, height * width, 3)

        # We reproject the points in world space
        world_points = cameras.unproject_points(ndc_points, scaled_depth_input=False)

        if features is not None:
            channels = features.shape[-1]
            point_features = 0. + features.view(batch_size, -1, channels)
            return world_points, point_features
        else:
            return world_points

    def warp(self, target_world_points,
             source_features, source_cameras,
             features_channel_is_at_the_end=False,
             mode='bilinear',
             resize_target_to_fit_source=True,
             padding_mode='zeros'):
        """

        :param target_world_points: Tensor with shape (n_img, target_height, target_width, 3)
        :param source_features: Tensor with shape (n_img, height, width, channels)
        OR (n_img, channels, height, width)
        :param source_cameras:
        :param features_channel_is_at_the_end:
        :param mode:
        :param resize_target_to_fit_source:
        :param padding_mode: (str) Can be either 'zeros', 'reflection' or 'border'.
        :return:
        """
        if features_channel_is_at_the_end:
            batch_size, height, width, channels = source_features.shape
        else:
            batch_size, channels, height, width = source_features.shape

        _, target_height, target_width, _ = target_world_points.shape

        # We project world points in ndc space
        screen_points = source_cameras.get_full_projection_transform().transform_points(
            target_world_points.view(batch_size, -1, 3), eps=1e-8)

        # We convert ndc coordinates to normalized screen coordinates
        factor = -1 * min(width, height)
        screen_points[..., 0] = factor / width * screen_points[..., 0]
        screen_points[..., 1] = factor / height * screen_points[..., 1]

        screen_points = screen_points[..., :2]
        screen_points = screen_points.view(batch_size, target_height, target_width, 2)

        # screen_points[..., [0,1]] = screen_points[..., [1,0]]

        if resize_target_to_fit_source:
            screen_points = transpose_channels(screen_points, channel_is_at_the_end=True)
            # screen_points = torchvision.transforms.Resize(size=(height, width))(screen_points)
            screen_points = torch.nn.functional.interpolate(screen_points, size=(height, width), mode='bicubic')
            screen_points = transpose_channels(screen_points, channel_is_at_the_end=False)

        if features_channel_is_at_the_end:
            input_features = transpose_channels(source_features, channel_is_at_the_end=True)
        else:
            input_features = source_features

        res = torch.nn.functional.grid_sample(input=input_features,
                                              grid=screen_points,
                                              mode=mode,
                                              padding_mode=padding_mode  # 'reflection', 'zeros'
                                              )

        if features_channel_is_at_the_end:
            res = transpose_channels(res, channel_is_at_the_end=False)

        return res

    def forward(self, x, R, T, zfar,
                x_alpha, R_alpha, T_alpha, zfar_alpha,
                device,
                return_cost_volume=False):
        """

        :param x: Tensor with shape (batch_size, feature_channels, feature_height, feature_width)
        :param R: Tensor with shape (batch_size, 3, 3)
        :param T: Tensor with shape (batch_size, 3)
        :param zfar: Tensor with shape (batch_size, )

        :param x_alpha: Tensor with shape (batch_size, n_alpha, feature_channels, feature_height, feature_width)
        :param R_alpha: Tensor with shape (batch_size, n_alpha, 3, 3)
        :param T_alpha: Tensor with shape (batch_size, n_alpha, 3)
        :param zfar_alpha: Tensor with shape (batch_size, n_alpha)
        :return:
        """

        if x.get_device() != self.x_tab.get_device():
            self.x_tab = self.x_tab.to(device)
            self.y_tab = self.y_tab.to(device)
            self.depth_bins = self.depth_bins.to(device)

        batch_size, n_alpha = x.shape[0], x_alpha.shape[1]

        # Initializing cameras
        cameras = FoVPerspectiveCameras(device=device,
                                        R=R.view(batch_size, 1, 3, 3
                                                 ).expand(-1, self.n_depth, -1, -1
                                                          ).contiguous().view(-1, 3, 3),
                                        T=T.view(batch_size, 1, 3
                                                 ).expand(-1, self.n_depth, -1
                                                          ).contiguous().view(-1, 3),
                                        zfar=zfar.view(batch_size, 1
                                                       ).expand(-1, self.n_depth
                                                                ).contiguous().view(-1)
                                        )

        cameras_alpha = FoVPerspectiveCameras(device=device,
                                              R=R_alpha.view(batch_size, 1, n_alpha, 3, 3
                                                             ).expand(-1, self.n_depth, -1, -1, -1
                                                                      ).contiguous().view(-1, 3, 3),
                                              T=T_alpha.view(batch_size, 1, n_alpha, 3
                                                             ).expand(-1, self.n_depth, -1, -1
                                                                      ).contiguous().view(-1, 3),
                                              zfar=zfar_alpha.view(batch_size, 1, n_alpha
                                                       ).expand(-1, self.n_depth, -1
                                                                ).contiguous().view(-1))

        # We reproject the target image x in 3D for n_depth depth planes
        depth_bins = self.depth_bins.view(1, -1, 1, 1, 1
                                          ).expand(batch_size, -1, self.height, self.width, -1)

        world_points = self.reproject_depth_map(depth=depth_bins.contiguous().view(-1, self.height, self.width, 1),
                                                cameras=cameras)

        # Then, we warped features from source images x_alpha to target camera
        warped_x = self.warp(target_world_points=world_points.view(batch_size, self.n_depth, 1,
                                                                   self.height, self.width, 3
                                                                   ).expand(-1, -1, n_alpha, -1, -1, -1
                                                                            ).contiguous().view(-1, self.height,
                                                                                                self.width, 3),
                             source_features=x_alpha.view(batch_size, 1, n_alpha,
                                                          self.feature_channels,
                                                          self.feature_height,
                                                          self.feature_width
                                                          ).expand(-1, self.n_depth, -1, -1, -1, -1
                                                                   ).contiguous().view(-1,
                                                                                       self.feature_channels,
                                                                                       self.feature_height,
                                                                                       self.feature_width),
                             source_cameras=cameras_alpha,
                             features_channel_is_at_the_end=False,
                             mode='bilinear',
                             resize_target_to_fit_source=True,
                             padding_mode='zeros')

        warped_x = warped_x.view(batch_size, self.n_depth, n_alpha,
                                 self.feature_channels, self.feature_height, self.feature_width)

        # We average on all alpha values
        warped_x = torch.mean(warped_x, dim=-4)

        # Finally, we compute the cost volume by taking the L1-distance of feature dimension
        cost_volume = torch.linalg.norm(warped_x - x.view(batch_size,
                                                          1,
                                                          self.feature_channels,
                                                          self.feature_height,
                                                          self.feature_width
                                                          ).expand(-1, self.n_depth, -1, -1, -1),
                                        dim=2, ord=1) / self.feature_channels

        res = torch.cat((x, cost_volume), dim=-3)
        res = self.relu(self.conv_reduce(res))

        if return_cost_volume:
            return res, cost_volume
        else:
            return res


class ExpansionLayer(nn.Module):
    def __init__(self, input_channels, inner_channels,
                 output_channels,
                 output_size,
                 additional_channels=None,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(ExpansionLayer, self).__init__()

        self.upconv = torch.nn.ConvTranspose2d(in_channels=input_channels,
                                               out_channels=inner_channels,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding)
        self.upelu = torch.nn.ELU()

        total_inner_channels = inner_channels
        if additional_channels is not None:
            total_inner_channels += additional_channels

        self.iconv = torch.nn.Conv2d(in_channels=total_inner_channels,
                                     out_channels=output_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     padding_mode='reflect')
        self.ielu = torch.nn.ELU()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.output_size = output_size
        self.inner_channels = inner_channels
        self.additional_channels = additional_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def upsample(self, x):
        return torch.nn.functional.interpolate(input=x,
                                               size=self.output_size,
                                               mode='nearest')

    def forward(self, x, x_add=None):
        # Apply upconvolution then upsampling
        res = self.upelu(self.upconv(x))
        res = self.upsample(res)

        # If a residual link is needed, we concatenate x_add to the result
        if (x_add is not None) and (self.additional_channels is not None):
            res = torch.cat((res, x_add), dim=-3)

        # We finally apply the last convolutional layer
        res = self.ielu(self.iconv(res))

        return res


class DisparityLayer(nn.Module):
    def __init__(self, input_channels):
        super(DisparityLayer, self).__init__()

        # Parameters
        self.channels = input_channels

        # Learnable layer
        self.conv = torch.nn.Conv2d(in_channels=input_channels,
                                    out_channels=1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    padding_mode='reflect')

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class DepthDecoder(nn.Module):
    def __init__(self,
                 feature_extractor,
                 resnet_model,
                 input_height=input_height, input_width=input_width, input_channels=input_channels,
                 n_alpha=n_alpha, d_min=d_min, d_max=d_max,
                 n_depth=n_depth,  # d_min and d_max correspond to znear and zfar
                 use_input_image_in_skip_connection=True
                 ):
        super(DepthDecoder, self).__init__()

        # -----Parameters-----
        self.height = input_height
        self.width = input_width
        self.channels = input_channels
        self.use_input_image_in_skip_connection = use_input_image_in_skip_connection

        # -----Feature extractor-----
        self.feature_extractor = feature_extractor

        # -----Cost Volume Builder-----
        self.cost_volume_builder = CostVolumeBuilder(height=input_height,
                                                     width=input_width,
                                                     feature_height=input_height//4,
                                                     feature_width=input_width//4,
                                                     feature_channels=64,
                                                     n_alpha=n_alpha,
                                                     d_min=d_min,
                                                     d_max=d_max,
                                                     n_depth=n_depth,
                                                     output_channels=64
                                                     )

        # -----Layers and learnable parameters-----

        # Contraction layers
        self.resnet_layer_2 = resnet_model.layer2
        self.resnet_layer_3 = resnet_model.layer3
        self.resnet_layer_4 = resnet_model.layer4

        # Expansion Layer 5
        self.expansion5 = ExpansionLayer(input_channels=512,
                                         inner_channels=256,
                                         output_channels=256,
                                         output_size=(input_height // 16,
                                                      input_width // 16 + (input_width % 16 > 0)),
                                         additional_channels=256)

        # Expansion Layer 4
        self.expansion4 = ExpansionLayer(input_channels=256,
                                         inner_channels=128,
                                         output_channels=128,
                                         output_size=(input_height // 8,
                                                      input_width // 8 + (input_width % 8 > 0)),
                                         additional_channels=128)
        self.disp4 = DisparityLayer(input_channels=128)

        # Expansion Layer 3
        self.expansion3 = ExpansionLayer(input_channels=128,
                                         inner_channels=64,
                                         output_channels=64,
                                         output_size=(input_height // 4,
                                                      input_width // 4 + (input_width % 4 > 0)),
                                         additional_channels=64)
        self.disp3 = DisparityLayer(input_channels=64)

        # Expansion Layer 2
        self.expansion2 = ExpansionLayer(input_channels=64,
                                         inner_channels=32,
                                         output_channels=32,
                                         output_size=(input_height // 2,
                                                      input_width // 2 + (input_width % 2 > 0)),
                                         additional_channels=64)
        self.disp2 = DisparityLayer(input_channels=32)

        # Expansion Layer 1
        last_expansion_additional_channels = None
        if use_input_image_in_skip_connection:
            last_expansion_additional_channels = 3

        self.expansion1 = ExpansionLayer(input_channels=32,
                                         inner_channels=16,
                                         output_channels=16,
                                         output_size=(input_height, input_width),
                                         additional_channels=last_expansion_additional_channels)
        self.disp1 = DisparityLayer(input_channels=16)

    def forward(self, x, R, T, zfar,
                x_alpha, R_alpha, T_alpha, zfar_alpha, device):
        """

        :param x: Tensor with shape (batch_size, channels, height, width)
        :param x_alpha: Tensor with shape (batch_size, n_alpha, channels, height, width)
        :return:
        """

        batch_size, n_alpha = x.shape[0], x_alpha.shape[1]

        # Extracting features from base image
        conv1 = self.feature_extractor.conv1(x)
        conv1 = self.feature_extractor.bn1(conv1)
        conv1 = self.feature_extractor.relu(conv1)

        maxpooled = self.feature_extractor.maxpool(conv1)
        layer1 = self.feature_extractor.layer(maxpooled)

        # Extracting features from additional images
        conv1_alpha = x_alpha.reshape(-1, self.channels, self.height, self.width)
        conv1_alpha = self.feature_extractor.conv1(conv1_alpha)
        conv1_alpha = self.feature_extractor.bn1(conv1_alpha)
        conv1_alpha = self.feature_extractor.relu(conv1_alpha)

        maxpooled_alpha = self.feature_extractor.maxpool(conv1_alpha)
        layer1_alpha = self.feature_extractor.layer(maxpooled_alpha)
        layer1_alpha = layer1_alpha.view(batch_size, n_alpha, 64, self.height//4, self.width//4)

        # Computing Cost-Volume
        conv_reduce = self.cost_volume_builder(layer1, R, T, zfar,
                                               layer1_alpha, R_alpha, T_alpha, zfar_alpha,
                                               device=device)

        # Contraction encoding
        layer2 = self.resnet_layer_2(conv_reduce)
        layer3 = self.resnet_layer_3(layer2)
        layer4 = self.resnet_layer_4(layer3)

        # Expansion decoding
        iconv5 = self.expansion5(x=layer4, x_add=layer3)

        iconv4 = self.expansion4(x=iconv5, x_add=layer2)
        disp4 = self.disp4(iconv4)

        iconv3 = self.expansion3(x=iconv4, x_add=layer1)
        disp3 = self.disp3(iconv3)

        iconv2 = self.expansion2(x=iconv3, x_add=conv1)
        disp2 = self.disp2(iconv2)

        last_x_add = None
        if self.use_input_image_in_skip_connection:
            last_x_add = x
        iconv1 = self.expansion1(x=iconv2, x_add=last_x_add)
        disp1 = self.disp1(iconv1)

        return disp1, disp2, disp3, disp4


def save_depth_decoder_from_resnet18(save_path, device, feature_extractor,
                                     input_height=input_height, input_width=input_width,
                                     input_channels=input_channels):
    # Load pretrained ResNet18 model
    resnet_model = torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=True).to(device)
    # resnet_model.eval()

    model = DepthDecoder(feature_extractor,
                         resnet_model,
                         input_height=input_height, input_width=input_width, input_channels=input_channels).to(device)

    # Save the whole model
    torch.save(model, save_path)


def load_depth_decoder(path='depth_decoder.pth', device=None):
    model = torch.load(path, map_location=device)
    return model


class PoseDecoder(nn.Module):
    def __init__(self,
                 resnet_model,
                 input_height=input_height, input_width=input_width
                 ):
        super(PoseDecoder, self).__init__()

        # -----Parameters-----
        self.height = input_height
        self.width = input_width

        # -----Layers and learnable parameters-----

        # Encoding layers
        conv1 = resnet_model.conv1
        self.biconv1 = torch.nn.Conv2d(in_channels=2 * conv1.in_channels,
                                       out_channels=conv1.out_channels,
                                       kernel_size=conv1.kernel_size,
                                       stride=conv1.stride,
                                       dilation=conv1.dilation,
                                       padding=conv1.padding,
                                       padding_mode=conv1.padding_mode,
                                       )
        for i in range(len(self.biconv1.weight)):
            self.biconv1.weight.data[i, :3] = conv1.weight.data[i] / 2.
            self.biconv1.weight.data[i, 3:] = conv1.weight.data[i] / 2.

        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool

        self.resnet_layer_1 = resnet_model.layer1
        self.resnet_layer_2 = resnet_model.layer2
        self.resnet_layer_3 = resnet_model.layer3
        self.resnet_layer_4 = resnet_model.layer4

        # Decoding layers
        self.pconv0 = torch.nn.Conv2d(in_channels=512,
                                      out_channels=256,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)  # Check the padding
        self.relu0 = torch.nn.ReLU()

        self.pconv1 = torch.nn.Conv2d(in_channels=256,
                                      out_channels=256,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.relu1 = torch.nn.ReLU()

        self.pconv2 = torch.nn.Conv2d(in_channels=256,
                                      out_channels=256,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.relu2 = torch.nn.ReLU()

        self.pconv3 = torch.nn.Conv2d(in_channels=256,
                                      out_channels=6,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        # Try to add another fc layer, like below
        self.fc = torch.nn.Linear(6 * input_height // 32 * (input_width // 32 + (input_width % 32 > 0)), 6)

        # self.fc1 = torch.nn.Linear(6 * input_height // 32 * (input_width // 32 + (input_width % 32 > 0)), 360)
        # self.fc_relu1 = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(360, 6)

    def forward(self, x, x_alpha):
        """

        :param x: Tensor with shape (batch_size, channels, height, width)
        :param x_alpha: Tensor with shape (batch_size, n_alpha, channels, height, width)
        :return:
        """

        batch_size, n_alpha = x.shape[0], x_alpha.shape[1]
        res = torch.cat((x.view(batch_size, 1, 3, self.height, self.width
                                ).expand(-1, n_alpha, -1, -1, -1
                                         ),
                         x_alpha),
                        dim=2
                        )
        res = res.view(-1, 6, self.height, self.width)
        # print("input:", torch.mean(res))

        res = self.biconv1(res)
        # print("biconv1:", torch.mean(res))
        res = self.bn1(res)
        # print("bn1:", torch.mean(res))
        res = self.relu(res)
        # print("relu:", torch.mean(res))
        res = self.maxpool(res)
        # print("maxpool:", torch.mean(res))

        res = self.resnet_layer_1(res)
        # print("resnet1:", torch.mean(res))
        res = self.resnet_layer_2(res)
        # print("resnet2:", torch.mean(res))
        res = self.resnet_layer_3(res)
        # print("resnet3:", torch.mean(res))
        res = self.resnet_layer_4(res)
        # print("resnet4:", torch.mean(res))

        res = self.relu0(self.pconv0(res))
        # print("pconv0:", torch.mean(res))
        res = self.relu1(self.pconv1(res))
        # print("pconv1:", torch.mean(res))
        res = self.relu2(self.pconv2(res))
        # print("pconv2:", torch.mean(res))

        res = self.pconv3(res)
        # print("pconv3:", torch.mean(res))
        res = res.view(batch_size, n_alpha, -1)

        res = self.fc(res)
        # print("fc:", torch.mean(res))

        # res = self.fc_relu1(self.fc1(res))
        # res = self.fc2(res)

        return res


def save_pose_decoder_from_resnet18(save_path, device, input_height=input_height, input_width=input_width):
    # Load pretrained ResNet18 model
    resnet_model = torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=True).to(device)
    # resnet_model.eval()

    model = PoseDecoder(resnet_model=resnet_model,
                        input_height=input_height,
                        input_width=input_width).to(device)

    # Save the whole model
    torch.save(model, save_path)


def load_pose_decoder(path='pose_decoder.pth', device=None):
    model = torch.load(path, map_location=device)
    return model


class ManyDepth(nn.Module):
    def __init__(self,
                 depth_decoder,
                 pose_decoder,
                 pose_factor=pose_factor,
                 learn_pose=learn_pose):
        super(ManyDepth, self).__init__()

        self.depth_decoder = depth_decoder
        if learn_pose:
            self.pose_decoder = pose_decoder
        self.pose_factor = pose_factor
        self.learn_pose = learn_pose

        self.input_height = depth_decoder.height
        self.input_width = depth_decoder.width

        self.d_min = depth_decoder.cost_volume_builder.d_min   
        self.d_max = depth_decoder.cost_volume_builder.d_max
        self.n_depth = depth_decoder.cost_volume_builder.n_depth

    def forward(self, x, x_alpha, R, T, zfar, device, gt_pose=None):
        batch_size, n_alpha = x.shape[0], x_alpha.shape[1]

        if self.d_max != zfar[0].item():
            raise NameError("Model variable d_max is different from the provided zfar.\n"
                            "Please check that d_min and d_max are respectively equal to parameters znear and zfar.")

        if self.learn_pose:
            pose = self.pose_decoder(x=x, x_alpha=x_alpha)
        else:
            if gt_pose is None:
                raise NameError("Input gt_pose is missing!"
                                "The parameter 'learn_pose' is set to False. "
                                "Consequently, Model must take ground truth poses as an input.")
            else:
                pose = gt_pose

        # CORRECT FORMULA:
        # recompose_R = R @ relative_R
        # recompose_T = relative_T + quaternion_apply(matrix_to_quaternion(relative_R.transpose(dim0=-1, dim1=-2)), T)

        relative_R = axis_angle_to_matrix(self.pose_factor * pose[..., 3:])
        relative_T = self.pose_factor * pose[..., :3]

        expanded_R = R.view(batch_size, 1, 3, 3).expand(-1, n_alpha, -1, -1)
        expanded_T = T.view(batch_size, 1, 3).expand(-1, n_alpha, -1)

        R_alpha = expanded_R @ relative_R
        T_alpha = relative_T + quaternion_apply(matrix_to_quaternion(relative_R.transpose(dim0=-1, dim1=-2)),
                                                expanded_T)

        zfar_alpha = zfar.view(batch_size, 1).expand(-1, n_alpha).contiguous()

        disp1, disp2, disp3, disp4 = self.depth_decoder(x=x, R=R, T=T,
                                                        zfar=zfar,
                                                        x_alpha=x_alpha, R_alpha=R_alpha, T_alpha=T_alpha,
                                                        zfar_alpha=zfar_alpha,
                                                        device=device)

        return pose, disp1, disp2, disp3, disp4


def create_many_depth_model(device, learn_pose=learn_pose,
                            pretrained_resnet_path="pretrained_resnet18.pth", save_resnet=False):
    if pretrained_resnet_path is None:
        resnet_model = torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=True).to(device)
        if save_resnet:
            torch.save(resnet_model.state_dict(), pretrained_resnet_path)
    else:
        resnet_model = torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=False).to(device)
        state_dict = torch.load(pretrained_resnet_path, map_location=device)
        resnet_model.load_state_dict(state_dict)
    resnet_model.eval()

    # Creates a feature extractor from the first layer
    feature_extractor = FeatureExtractor(resnet_model).to(device)

    depth_decoder = DepthDecoder(feature_extractor,
                                 resnet_model,
                                 input_height=input_height, input_width=input_width, input_channels=input_channels
                                 ).to(device)

    if learn_pose:
        if pretrained_resnet_path is None:
            resnet_model_2 = torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=True).to(device)
        else:
            resnet_model_2 = torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=False).to(device)
            state_dict = torch.load(pretrained_resnet_path, map_location=device)
            resnet_model_2.load_state_dict(state_dict)
        pose_decoder = PoseDecoder(resnet_model=resnet_model_2,
                                   input_height=input_height,
                                   input_width=input_width).to(device)
    else:
        pose_decoder = None

    model = ManyDepth(depth_decoder=depth_decoder, pose_decoder=pose_decoder).to(device)

    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("input_height:", model.input_height)
    print("input_width:", model.input_width)
    print("n_alpha:", model.depth_decoder.cost_volume_builder.n_alpha)
    print("d_min:", model.d_min)
    print("d_max:", model.d_max)
    print("n_depth:", model.n_depth)
    print("pose_factor:", model.pose_factor)
    print("learn_pose:", model.learn_pose)

    return model


class SSIM(nn.Module):
    """
    Layer to compute the SSIM loss between a pair of images
    Credits to https://github.com/nianticlabs/manydepth/blob/master/manydepth/
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
