from typing import List, Tuple, Union

from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import Tensor, nn

class _TrafficSignifierConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], **kwargs) -> None:
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, bias=True, **kwargs)
        self.bnorm = nn.BatchNorm2d(
            num_features=out_channels, eps=1e-4, momentum=1e-1, affine=True)
        self.gelu = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2d(x)
        x = self.bnorm(x)
        x = self.gelu(x)
        return x

class _TrafficSignifierPooledConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]]) -> None:
        super().__init__()

        self.pooling = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)
        self.conv = _TrafficSignifierConv2d(
            in_channels=in_channels - 2, # Result of Average Pooling
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pooling(x)
        x = self.conv(x)
        return x

class _TrafficSignifierLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, growth_rate: int, kernel_size: Union[int, Tuple[int, int]], dropout_rate: float) -> None:
        super().__init__()

        self.conv2d_1x1 = _TrafficSignifierConv2d(
            in_channels=in_channels,
            out_channels=out_channels * growth_rate,
            kernel_size=1
        )

        self.conv2d_nxn = _TrafficSignifierConv2d(
            in_channels=out_channels * growth_rate,
            out_channels=out_channels,
            kernel_size=kernel_size
        )

        self.dropout_rate = float(dropout_rate)

    def forward(self, x: Union[Tensor, List[Tensor]]) -> Tensor:
        if isinstance(x, Tensor):
            x = [x]

        x = torch.cat(x, dim=1)

        x = self.conv2d_1x1(x)
        x = self.conv2d_nxn(x)

        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x

class _TrafficSignifierBlock(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, pool_out_channels: int, growth_rate: int, dropout_rate: float) -> None:
        super().__init__()

        self.layer_1x1 = _TrafficSignifierConv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1,
        )

        self.growing_layer_3x3 = nn.ModuleDict()
        for i in range(num_layers):
            self.growing_layer_3x3.add_module(f"layer{i + 1}_3x3", _TrafficSignifierLayer(
                in_channels=in_channels + i * 32,
                out_channels=32,
                kernel_size=3,
                growth_rate=(i+1) * growth_rate,
                dropout_rate=dropout_rate
            ))

        self.pooled_layer_1x1 = _TrafficSignifierPooledConv2d(
            in_channels=in_channels,
            out_channels=pool_out_channels,
            kernel_size=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_layer_1x1 = self.layer_1x1(x)

        x_growing_layer = [x]
        for _, layer in self.growing_layer_3x3.items():
            x_growing_layer.append(layer(x_growing_layer))

        x_pooled_1x1 = self.pooled_layer_1x1(x)

        outputs = [x_growing_layer[0], x_layer_1x1]
        outputs.extend(x_growing_layer[1:])
        outputs.append(x_pooled_1x1)

        return torch.cat(outputs, dim=1)

class TrafficSignifier(nn.Module):
    def __init__(self, num_classes: int = 4, num_blocks: int = 4, num_internal_layers: int = 3, dropout_rate: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict([
                ("conv0", _TrafficSignifierConv2d(
                    in_channels=3,
                    out_channels=32,
                    kernel_size=3
                )),
                ("conv1", _TrafficSignifierConv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3
                )),
                ("conv2", _TrafficSignifierConv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=2
                )),
                ("gelu0", nn.GELU()),
                ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ])
        )

        input_size = 64
        for i in range(num_blocks):
            pool_out_channels = 32 * (i + 1)
            self.features.add_module(f"trafficsignifierblock{i+1}", _TrafficSignifierBlock(
                in_channels=input_size,
                num_layers=num_internal_layers,
                pool_out_channels=pool_out_channels,
                growth_rate=2,
                dropout_rate=dropout_rate
            ))

            input_size += 64 + pool_out_channels + 32 * num_internal_layers

        self.features_gelu = nn.GELU()
        self.adapt_avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_features=input_size, out_features=num_classes, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.features_gelu(x)
        x = self.adapt_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x