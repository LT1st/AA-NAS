"""
Copyright (C) 2017, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import logging
from collections import OrderedDict

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models.densenet as _model
from torchvision.models.densenet import _DenseBlock, _Transition, model_urls

import model


class DenseNet(_model.DenseNet):
    def __init__(self, config_channels, anchors, num_cls, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0):
        nn.Module.__init__(self)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('conv', nn.Conv2d(num_features, model.output_channels(len(anchors), num_cls), 1))

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.features(x)


def densenet121(config_channels, anchors, num_cls, **kwargs):
    model = DenseNet(config_channels, anchors, num_cls, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    if config_channels.config.getboolean('model', 'pretrained'):
        url = model_urls['densenet121']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def densenet169(config_channels, anchors, num_cls, **kwargs):
    model = DenseNet(config_channels, anchors, num_cls, num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    if config_channels.config.getboolean('model', 'pretrained'):
        url = model_urls['densenet169']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def densenet201(config_channels, anchors, num_cls, **kwargs):
    model = DenseNet(config_channels, anchors, num_cls, num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    if config_channels.config.getboolean('model', 'pretrained'):
        url = model_urls['densenet201']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model


def densenet161(config_channels, anchors, num_cls, **kwargs):
    model = DenseNet(config_channels, anchors, num_cls, num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs)
    if config_channels.config.getboolean('model', 'pretrained'):
        url = model_urls['densenet161']
        logging.info('use pretrained model: ' + url)
        state_dict = model.state_dict()
        for key, value in model_zoo.load_url(url).items():
            if key in state_dict:
                state_dict[key] = value
        model.load_state_dict(state_dict)
    return model
