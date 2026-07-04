#!/usr/bin/env python3
"""
Custom Spiking ResNet Variants for Parameter Optimization

Three optimized variants of Spiking ResNet-18:
- Variant A: Remove layer4 (3-stage, ~2.78M params)
- Variant B: Thin 4-stage (half channels, ~2.79M params)
- Variant C: Ultra-light (thin 3-stage, ~0.70M params)

Uses SpikingJelly's BasicBlock and spiking layer primitives.
"""

import torch
import torch.nn as nn
from copy import deepcopy

from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly.activation_based.model.spiking_resnet import BasicBlock


class CustomSpikingResNet(nn.Module):
    """
    Custom Spiking ResNet with configurable channel widths and number of stages.

    Mirrors SpikingJelly's SpikingResNet but allows:
    - Arbitrary initial channel count (stem conv output)
    - Arbitrary stage channel widths
    - Variable number of stages (3 or 4)
    """

    def __init__(self, block, layers, stage_channels, initial_channels=64,
                 num_classes=1000, zero_init_residual=False,
                 norm_layer=None, spiking_neuron: callable = None, **kwargs):
        """
        Args:
            block: Block type (BasicBlock)
            layers: List of block counts per stage, e.g. [2, 2, 2] or [2, 2, 2, 2]
            stage_channels: List of channel widths per stage, e.g. [64, 128, 256]
            initial_channels: Output channels of stem conv (default: 64)
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
            f"layers ({len(layers)}) and stage_channels ({len(stage_channels)}) must have same length"

        self.inplanes = initial_channels
        self.num_stages = len(layers)

        # Stem: Conv7x7 -> BN -> SN -> MaxPool
        self.conv1 = layer.Conv2d(3, initial_channels, kernel_size=7, stride=2,
                                  padding=3, bias=False)
        self.bn1 = norm_layer(initial_channels)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)

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

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)

        for stage in self.stages:
            x = stage(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


# ============================================================================
# Variant Configurations
# ============================================================================

VARIANT_CONFIGS = {
    'A': {
        'name': 'Variant A (No Layer4)',
        'description': 'Remove layer4, keep original channel widths [64, 128, 256]',
        'initial_channels': 64,
        'stage_channels': [64, 128, 256],
        'layers': [2, 2, 2],
    },
    'B': {
        'name': 'Variant B (Thin 4-Stage)',
        'description': 'Keep 4 stages, half channel widths [32, 64, 128, 256]',
        'initial_channels': 32,
        'stage_channels': [32, 64, 128, 256],
        'layers': [2, 2, 2, 2],
    },
    'C': {
        'name': 'Variant C (Ultra-Light)',
        'description': 'Remove layer4 + half channels [32, 64, 128]',
        'initial_channels': 32,
        'stage_channels': [32, 64, 128],
        'layers': [2, 2, 2],
    },
}


class SpikingResNetOptimized(nn.Module):
    """
    Optimized Spiking ResNet for crack detection with dropout and regularization.
    Drop-in replacement for SpikingResNetCrackDetector with configurable architecture.

    Supports 3 variants: A, B, C (see VARIANT_CONFIGS).
    Interface is identical to SpikingResNetCrackDetector for compatibility.
    """

    def __init__(self,
                 variant: str = 'A',
                 spiking_neuron: callable = neuron.IFNode,
                 surrogate_function: callable = surrogate.ATan(),
                 detach_reset: bool = True,
                 num_classes: int = 2,
                 zero_init_residual: bool = False,
                 T: int = 4,
                 dropout_rate: float = 0.0):
        super().__init__()

        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"Unknown variant '{variant}'. Must be one of: {list(VARIANT_CONFIGS.keys())}")

        self.variant = variant
        self.variant_config = VARIANT_CONFIGS[variant]
        self.T = T
        self.dropout_rate = dropout_rate

        # Build custom spiking ResNet based on variant config
        self.model = CustomSpikingResNet(
            block=BasicBlock,
            layers=self.variant_config['layers'],
            stage_channels=self.variant_config['stage_channels'],
            initial_channels=self.variant_config['initial_channels'],
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            spiking_neuron=spiking_neuron,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
        )

        # Add dropout layers if specified
        if dropout_rate > 0.0:
            if hasattr(self.model, 'fc'):
                in_features = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes)
                )

        # Store neuron references for regularization
        self.spiking_neurons = []
        self._collect_spiking_neurons()

    def _collect_spiking_neurons(self):
        """Collect all spiking neurons for regularization"""
        for module in self.model.modules():
            if isinstance(module, neuron.BaseNode):
                self.spiking_neurons.append(module)

    def forward(self, x):
        # x shape: [N, C, H, W]
        # Repeat input for T timesteps
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, N, C, H, W]

        # Process through time steps
        out_spikes = []
        for t in range(self.T):
            out = self.model(x_seq[t])
            out_spikes.append(out)

        # Aggregate spikes over time
        out = torch.stack(out_spikes, dim=0).mean(dim=0)  # [N, num_classes]

        # Reset the network state
        functional.reset_net(self.model)

        return out

    def get_membrane_regularization(self):
        """Calculate membrane potential regularization loss"""
        membrane_loss = 0.0
        count = 0

        for neuron_module in self.spiking_neurons:
            if hasattr(neuron_module, 'v') and neuron_module.v is not None:
                v = neuron_module.v
                if isinstance(v, (int, float)):
                    v = torch.tensor(v, device=next(self.parameters()).device)
                elif isinstance(v, torch.Tensor) and v.numel() > 0:
                    membrane_loss += torch.mean(v ** 2)
                    count += 1

        return membrane_loss / max(count, 1) if count > 0 else torch.tensor(0.0, device=next(self.parameters()).device)

    def get_synaptic_regularization(self):
        """Calculate synaptic strength regularization loss"""
        synaptic_loss = 0.0
        count = 0

        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                synaptic_loss += torch.norm(param, 2)
                count += 1

        return synaptic_loss / max(count, 1) if count > 0 else torch.tensor(0.0, device=next(self.parameters()).device)

    def get_spike_rate_regularization(self):
        """Calculate spike rate regularization to prevent over-spiking"""
        spike_rate_loss = 0.0
        count = 0

        for neuron_module in self.spiking_neurons:
            if hasattr(neuron_module, 'spike') and neuron_module.spike is not None:
                spike = neuron_module.spike
                if isinstance(spike, (int, float)):
                    spike = torch.tensor(spike, device=next(self.parameters()).device)
                elif isinstance(spike, torch.Tensor) and spike.numel() > 0:
                    spike_rate = torch.mean(spike.float())
                    target_rate = 0.2
                    spike_rate_loss += (spike_rate - target_rate) ** 2
                    count += 1

        return spike_rate_loss / max(count, 1) if count > 0 else torch.tensor(0.0, device=next(self.parameters()).device)

    def get_variant_info(self):
        """Return information about the current variant"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)

        return {
            'variant': self.variant,
            'name': self.variant_config['name'],
            'description': self.variant_config['description'],
            'initial_channels': self.variant_config['initial_channels'],
            'stage_channels': self.variant_config['stage_channels'],
            'layers': self.variant_config['layers'],
            'num_stages': len(self.variant_config['layers']),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'time_steps': self.T,
            'dropout_rate': self.dropout_rate,
        }


def create_variant(variant: str, num_classes: int = 2, T: int = 4,
                   dropout_rate: float = 0.0, **kwargs) -> SpikingResNetOptimized:
    """
    Factory function to create a specific variant.

    Args:
        variant: 'A', 'B', or 'C'
        num_classes: Number of output classes
        T: Number of time steps
        dropout_rate: Dropout rate

    Returns:
        SpikingResNetOptimized model
    """
    return SpikingResNetOptimized(
        variant=variant,
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        num_classes=num_classes,
        T=T,
        dropout_rate=dropout_rate,
    )


# ============================================================================
# Self-test: verify parameter counts
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Custom Spiking ResNet Variants - Parameter Analysis")
    print("=" * 70)

    # Also show original for comparison
    from spikingjelly.activation_based.model import spiking_resnet
    original = spiking_resnet.spiking_resnet18(
        pretrained=False,
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        num_classes=2
    )
    orig_params = sum(p.numel() for p in original.parameters())
    print(f"\nOriginal Spiking ResNet-18:")
    print(f"  Channels: [64, 128, 256, 512], 4 stages, layers=[2,2,2,2]")
    print(f"  Parameters: {orig_params:,} ({orig_params/1e6:.2f}M)")
    print(f"  Model size: {orig_params * 4 / (1024*1024):.2f} MB")

    print()

    for variant_key in ['A', 'B', 'C']:
        model = create_variant(variant_key, num_classes=2, T=4)
        info = model.get_variant_info()

        reduction = (1 - info['total_parameters'] / orig_params) * 100

        print(f"{info['name']}:")
        print(f"  {info['description']}")
        print(f"  Channels: {info['stage_channels']}, {info['num_stages']} stages, layers={info['layers']}")
        print(f"  Parameters: {info['total_parameters']:,} ({info['total_parameters']/1e6:.2f}M)")
        print(f"  Model size: {info['model_size_mb']:.2f} MB")
        print(f"  Reduction: {reduction:.1f}%")

        # Quick forward pass test
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
        print(f"  Forward pass: input={list(x.shape)} -> output={list(y.shape)}")
        print()

    print("All variants verified successfully!")
