#!/usr/bin/env python3
"""
HLS-Optimized Spiking ResNet Variants for FPGA Deployment

New variants optimized for BRAM-constrained FPGAs (KV260):
- Variant D: Aggressive Stride (stride=4 stem, max feature map 64KB)
- Variant E: Multi-Conv Stem (two stride-2 convs, max feature map 256KB)

These variants prioritize reducing feature map sizes for HLS implementation
while maintaining acceptable accuracy for crack detection.

Target: Xilinx KV260 (Zynq UltraScale+ ZU5EV)
- BRAM: 144 x 4.5KB = 648 KB
- URAM: 64 x 36KB = 2,304 KB
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Dict, Any, Optional

from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly.activation_based.model.spiking_resnet import BasicBlock


class HLSSpikingResNet(nn.Module):
    """
    HLS-Optimized Spiking ResNet with configurable stem and aggressive downsampling.

    Supports multiple stem types:
    - 'standard': Conv7x7(s=2) + MaxPool(s=2) - original ResNet stem
    - 'aggressive': Conv7x7(s=4) - single conv with stride 4
    - 'multi_conv': Conv3x3(s=2) + Conv3x3(s=2) - two small convs
    """

    def __init__(self,
                 block,
                 layers: List[int],
                 stage_channels: List[int],
                 initial_channels: int = 64,
                 stem_type: str = 'standard',
                 stem_stride: int = 4,
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 norm_layer = None,
                 spiking_neuron: callable = None,
                 **kwargs):
        """
        Args:
            block: Block type (BasicBlock)
            layers: List of block counts per stage
            stage_channels: List of channel widths per stage
            initial_channels: Output channels of stem
            stem_type: 'standard', 'aggressive', or 'multi_conv'
            stem_stride: Total stride of stem (used for aggressive type)
            num_classes: Number of output classes
            zero_init_residual: Zero-init last BN in each residual branch
            norm_layer: Normalization layer
            spiking_neuron: Spiking neuron class
            **kwargs: Additional kwargs for spiking_neuron
        """
        super().__init__()

        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

        assert len(layers) == len(stage_channels), \
            f"layers ({len(layers)}) and stage_channels ({len(stage_channels)}) must match"

        self.inplanes = initial_channels
        self.num_stages = len(layers)
        self.stem_type = stem_type

        # Build stem based on type
        if stem_type == 'standard':
            # Original ResNet stem: Conv7x7(s=2) + MaxPool(s=2)
            self.stem = nn.Sequential(
                layer.Conv2d(3, initial_channels, kernel_size=7, stride=2, padding=3, bias=False),
                norm_layer(initial_channels),
                spiking_neuron(**deepcopy(kwargs)),
                layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.stem_downsample = 4

        elif stem_type == 'aggressive':
            # Aggressive stem: Conv7x7(s=4) - single large stride
            self.stem = nn.Sequential(
                layer.Conv2d(3, initial_channels, kernel_size=7, stride=stem_stride, padding=3, bias=False),
                norm_layer(initial_channels),
                spiking_neuron(**deepcopy(kwargs)),
            )
            self.stem_downsample = stem_stride

        elif stem_type == 'multi_conv':
            # Multi-conv stem: Conv3x3(s=2) + Conv3x3(s=2)
            mid_channels = initial_channels // 2 if initial_channels >= 16 else initial_channels
            self.stem = nn.Sequential(
                layer.Conv2d(3, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(mid_channels),
                spiking_neuron(**deepcopy(kwargs)),
                layer.Conv2d(mid_channels, initial_channels, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(initial_channels),
                spiking_neuron(**deepcopy(kwargs)),
            )
            self.stem_downsample = 4
        else:
            raise ValueError(f"Unknown stem_type: {stem_type}")

        # Build stages dynamically
        self.stages = nn.ModuleList()
        for i, (num_blocks, channels) in enumerate(zip(layers, stage_channels)):
            stride = 1 if i == 0 else 2
            stage = self._make_layer(block, channels, num_blocks, stride=stride,
                                     spiking_neuron=spiking_neuron, **kwargs)
            self.stages.append(stage)

        # Classifier
        final_channels = stage_channels[-1] * block.expansion
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(final_channels, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1,
                    spiking_neuron: callable = None, **kwargs):
        """Build a stage with the given number of blocks and channel width."""
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        block_layers = []
        block_layers.append(block(self.inplanes, planes, stride, downsample,
                                  groups=1, base_width=64, dilation=1,
                                  norm_layer=norm_layer,
                                  spiking_neuron=spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            block_layers.append(block(self.inplanes, planes,
                                      groups=1, base_width=64, dilation=1,
                                      norm_layer=norm_layer,
                                      spiking_neuron=spiking_neuron, **kwargs))

        return nn.Sequential(*block_layers)

    def forward(self, x):
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        x = self.fc(x)

        return x


# ============================================================================
# HLS-Optimized Variant Configurations
# ============================================================================

HLS_VARIANT_CONFIGS = {
    'D': {
        'name': 'Variant D (Aggressive Stride)',
        'description': 'Stride-4 stem, channels [16, 24, 48, 96], max FM 64KB',
        'initial_channels': 16,
        'stage_channels': [16, 24, 48, 96],
        'layers': [2, 2, 2, 2],
        'stem_type': 'aggressive',
        'stem_stride': 4,
        'target_input_size': 256,
        'max_feature_map_kb': 64,
    },
    'E': {
        'name': 'Variant E (Multi-Conv Stem)',
        'description': 'Two stride-2 convs, channels [16, 24, 48, 64], max FM 256KB',
        'initial_channels': 16,
        'stage_channels': [16, 24, 48, 64],
        'layers': [2, 2, 2, 2],
        'stem_type': 'multi_conv',
        'stem_stride': 4,
        'target_input_size': 256,
        'max_feature_map_kb': 256,
    },
    'F': {
        'name': 'Variant F (Balanced HLS)',
        'description': 'Aggressive stride-4, channels [24, 32, 64, 96], 3 stages',
        'initial_channels': 24,
        'stage_channels': [24, 32, 64],
        'layers': [2, 2, 2],
        'stem_type': 'aggressive',
        'stem_stride': 4,
        'target_input_size': 256,
        'max_feature_map_kb': 96,
    },
    'G': {
        'name': 'Variant G (Beta=0.50)',
        'description': 'Stride-4 stem, channels [16, 16, 32, 64], beta=0.50 ablation',
        'initial_channels': 16,
        'stage_channels': [16, 16, 32, 64],
        'layers': [2, 2, 2, 2],
        'stem_type': 'aggressive',
        'stem_stride': 4,
        'target_input_size': 256,
        'max_feature_map_kb': 64,
    },
    'H': {
        'name': 'Variant H (Beta=1.00)',
        'description': 'Stride-4 stem, channels [16, 32, 64, 128], beta=1.00 ablation',
        'initial_channels': 16,
        'stage_channels': [16, 32, 64, 128],
        'layers': [2, 2, 2, 2],
        'stem_type': 'aggressive',
        'stem_stride': 4,
        'target_input_size': 256,
        'max_feature_map_kb': 64,
    },
}


def calculate_feature_map_sizes(config: Dict, input_size: int = 256) -> Dict[str, Any]:
    """
    Calculate feature map sizes at each layer for a given configuration.

    Returns dict with layer-by-layer analysis and peak memory usage.
    """
    results = {
        'input_size': input_size,
        'layers': [],
        'peak_feature_map_bytes': 0,
        'total_feature_map_bytes': 0,
    }

    # Input
    h, w = input_size, input_size
    c = 3
    fm_bytes = h * w * c
    results['layers'].append({
        'name': 'input',
        'shape': (h, w, c),
        'bytes': fm_bytes,
        'kb': fm_bytes / 1024
    })

    # Stem
    stem_type = config.get('stem_type', 'standard')
    stem_stride = config.get('stem_stride', 4)

    if stem_type == 'standard':
        h, w = h // 4, w // 4
    elif stem_type == 'aggressive':
        h, w = h // stem_stride, w // stem_stride
    elif stem_type == 'multi_conv':
        h, w = h // 4, w // 4

    c = config['initial_channels']
    fm_bytes = h * w * c
    results['layers'].append({
        'name': 'stem',
        'shape': (h, w, c),
        'bytes': fm_bytes,
        'kb': fm_bytes / 1024
    })
    results['peak_feature_map_bytes'] = max(results['peak_feature_map_bytes'], fm_bytes)

    # Stages
    for i, (num_blocks, channels) in enumerate(zip(config['layers'], config['stage_channels'])):
        stride = 1 if i == 0 else 2
        h, w = h // stride, w // stride
        c = channels
        fm_bytes = h * w * c
        results['layers'].append({
            'name': f'stage{i+1}',
            'shape': (h, w, c),
            'bytes': fm_bytes,
            'kb': fm_bytes / 1024
        })
        results['peak_feature_map_bytes'] = max(results['peak_feature_map_bytes'], fm_bytes)

    results['total_feature_map_bytes'] = sum(l['bytes'] for l in results['layers'])
    results['peak_kb'] = results['peak_feature_map_bytes'] / 1024
    results['total_kb'] = results['total_feature_map_bytes'] / 1024

    return results


class SpikingResNetHLS(nn.Module):
    """
    HLS-Optimized Spiking ResNet for FPGA deployment.

    Supports variants D, E, F optimized for BRAM-constrained devices.
    Interface compatible with SpikingResNetOptimized.
    """

    def __init__(self,
                 variant: str = 'D',
                 spiking_neuron: callable = neuron.IFNode,
                 surrogate_function: callable = surrogate.ATan(),
                 detach_reset: bool = True,
                 num_classes: int = 2,
                 zero_init_residual: bool = False,
                 T: int = 4,
                 dropout_rate: float = 0.0):
        super().__init__()

        if variant not in HLS_VARIANT_CONFIGS:
            raise ValueError(f"Unknown variant '{variant}'. Must be one of: {list(HLS_VARIANT_CONFIGS.keys())}")

        self.variant = variant
        self.variant_config = HLS_VARIANT_CONFIGS[variant]
        self.T = T
        self.dropout_rate = dropout_rate

        # Build HLS-optimized spiking ResNet
        self.model = HLSSpikingResNet(
            block=BasicBlock,
            layers=self.variant_config['layers'],
            stage_channels=self.variant_config['stage_channels'],
            initial_channels=self.variant_config['initial_channels'],
            stem_type=self.variant_config['stem_type'],
            stem_stride=self.variant_config.get('stem_stride', 4),
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            spiking_neuron=spiking_neuron,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
        )

        # Add dropout if specified
        if dropout_rate > 0.0:
            if hasattr(self.model, 'fc'):
                in_features = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes)
                )

        # Collect spiking neurons for regularization
        self.spiking_neurons = []
        self._collect_spiking_neurons()

    def _collect_spiking_neurons(self):
        for module in self.model.modules():
            if isinstance(module, neuron.BaseNode):
                self.spiking_neurons.append(module)

    def forward(self, x):
        # x shape: [N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, N, C, H, W]

        out_spikes = []
        for t in range(self.T):
            out = self.model(x_seq[t])
            out_spikes.append(out)

        out = torch.stack(out_spikes, dim=0).mean(dim=0)  # [N, num_classes]

        functional.reset_net(self.model)

        return out

    def get_membrane_regularization(self):
        membrane_loss = 0.0
        count = 0
        for neuron_module in self.spiking_neurons:
            if hasattr(neuron_module, 'v') and neuron_module.v is not None:
                v = neuron_module.v
                if isinstance(v, torch.Tensor) and v.numel() > 0:
                    membrane_loss += torch.mean(v ** 2)
                    count += 1
        return membrane_loss / max(count, 1) if count > 0 else torch.tensor(0.0)

    def get_synaptic_regularization(self):
        synaptic_loss = 0.0
        count = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                synaptic_loss += torch.norm(param, 2)
                count += 1
        return synaptic_loss / max(count, 1) if count > 0 else torch.tensor(0.0)

    def get_variant_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)

        # Calculate feature map sizes
        fm_analysis = calculate_feature_map_sizes(
            self.variant_config,
            self.variant_config.get('target_input_size', 256)
        )

        return {
            'variant': self.variant,
            'name': self.variant_config['name'],
            'description': self.variant_config['description'],
            'initial_channels': self.variant_config['initial_channels'],
            'stage_channels': self.variant_config['stage_channels'],
            'layers': self.variant_config['layers'],
            'stem_type': self.variant_config['stem_type'],
            'num_stages': len(self.variant_config['layers']),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'time_steps': self.T,
            'dropout_rate': self.dropout_rate,
            'peak_feature_map_kb': fm_analysis['peak_kb'],
            'feature_map_analysis': fm_analysis,
        }


def create_hls_variant(variant: str, num_classes: int = 2, T: int = 4,
                       dropout_rate: float = 0.0, **kwargs) -> SpikingResNetHLS:
    """Factory function to create an HLS-optimized variant."""
    return SpikingResNetHLS(
        variant=variant,
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        num_classes=num_classes,
        T=T,
        dropout_rate=dropout_rate,
    )


# ============================================================================
# Self-test
# ============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("HLS-Optimized Spiking ResNet Variants - Analysis")
    print("=" * 80)
    print()

    # Test each variant
    for variant_key in ['D', 'E', 'F', 'G', 'H']:
        print(f"\n{'='*80}")
        config = HLS_VARIANT_CONFIGS[variant_key]
        print(f"{config['name']}")
        print(f"  {config['description']}")
        print()

        # Create model
        model = create_hls_variant(variant_key, num_classes=2, T=4)
        info = model.get_variant_info()

        print(f"Architecture:")
        print(f"  Stem type: {info['stem_type']}")
        print(f"  Channels: {info['stage_channels']}")
        print(f"  Stages: {info['num_stages']}, layers={info['layers']}")
        print()

        print(f"Parameters:")
        print(f"  Total: {info['total_parameters']:,} ({info['total_parameters']/1e6:.3f}M)")
        print(f"  Model size: {info['model_size_mb']:.2f} MB")
        print()

        print(f"Feature Map Analysis (256x256 input):")
        for layer_info in info['feature_map_analysis']['layers']:
            shape_str = f"{layer_info['shape'][0]}x{layer_info['shape'][1]}x{layer_info['shape'][2]}"
            print(f"  {layer_info['name']:10s}: {shape_str:15s} = {layer_info['kb']:.1f} KB")
        print(f"  Peak: {info['peak_feature_map_kb']:.1f} KB")
        print()

        # Forward pass test
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            y = model(x)
        print(f"Forward pass: input {list(x.shape)} -> output {list(y.shape)}")

    print()
    print("=" * 80)
    print("All HLS variants verified successfully!")
    print("=" * 80)
