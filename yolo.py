import utils
import models
import constants
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()
        self.config_list = utils.parse_config()
        self.module_list = models.create_modules()

    def transform_output(self, output, input_dim, anchors, num_classes, CUDA=False):
        batch_size = output.size(0)
        stride = input_dim // output.size(2)
        # print(input_dim, stride)
        grid_size = input_dim // stride
        bbox_attrs = 5 + num_classes
        num_anchors = len(anchors)
        anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

        # Transform output from 3D [x, y, (5+C)*B] => 2D [x*y*B, 5+C]
        # for easier processing of different scale grids
        # print(output.shape)
        # print(batch_size, bbox_attrs, num_anchors, grid_size, grid_size)
        # print(batch_size * bbox_attrs * num_anchors * grid_size * grid_size)
        output = output.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

        # Sigmoid of centre coordinates and object confidence
        output[:, :, 0] = torch.sigmoid(output[:, :, 0])
        output[:, :, 1] = torch.sigmoid(output[:, :, 1])
        output[:, :, 4] = torch.sigmoid(output[:, :, 4])

        # Add the center offsets
        grid = np.arange(grid_size)
        a, b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1, 1)
        y_offset = torch.FloatTensor(b).view(-1, 1)

        if CUDA:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

        output[:, :, :2] += x_y_offset

        # log space transform for height and width
        anchors = torch.FloatTensor(anchors)

        if CUDA:
            anchors = anchors.cuda()

        anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
        output[:, :, 2:4] = torch.exp(output[:, :, 2:4]) * anchors

        # Sigmoid of class scores
        output[:, :, 5: 5 + num_classes] = torch.sigmoid((output[:, :, 5: 5 + num_classes]))

        # Resize bounding box properties w.r.t input image dimensions using stride
        output[:, :, :4] *= stride

        return output


    def forward(self, input, CUDA):
        output_cache = {}
        net_properties = self.config_list[0]
        detections_list = []
        for (index, (module_config, module)) in enumerate(zip(self.config_list[1:], self.module_list)):
            module_type = module_config["type"]
            print(module_type)
            if module_type in [constants.CONVOLUTIONAL_LAYER, constants.UPSAMPLE_LAYER]:
                output = module(input)
            elif module_type == constants.SHORTCUT_LAYER:
                shortcut_from = int(module_config["from"])
                if shortcut_from > 0:
                    shortcut_from -= index
                output = output_cache[index - 1] + output_cache[index + shortcut_from]
            elif module_type == constants.ROUTE_LAYER:
                layers = module_config["layers"].split(',')
                layer_list = []
                for layer in layers:
                    layer_index = int(layer)
                    if layer_index > 0:
                        layer_index -= index
                    layer_list.append(output_cache[index + layer_index])
                output = torch.cat(layer_list, dim=1)
            elif module_type == constants.YOLO_LAYER:
                anchors = module[0].anchors
                input_dim = int(net_properties["height"])
                num_classes = int(module_config["classes"])
                output = self.transform_output(input, input_dim, anchors, num_classes, CUDA)
                detections_list.append(output)
                print(output.shape)

            output_cache[index] = output
            input = output
        detections = torch.cat(detections_list, 1)
        return detections