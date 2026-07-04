# SNN-FPGA-SHM: Spiking Neural Network on FPGA for Structural Health Monitoring

A resource-aware co-design framework for deploying Spiking Neural Networks (SNNs) on edge FPGAs for concrete crack detection. The complete pipeline covers model training, INT8 post-training quantization, HLS hardware synthesis, and physical deployment on the Xilinx Kria KV260 platform.

## Key Results

| Metric | Value |
|--------|-------|
| Platform | Xilinx Kria KV260 (Zynq UltraScale+ ZU5EV) |
| Accuracy | 86.4% (FP32) / 86.3% (INT8) |
| Memory | 245 BRAM (85% reduction from baseline 865) |
| DSP | 55 (87% reduction from baseline 426) |
| Dynamic Power | 0.321 W (board-measured) |
| Clock | 174 MHz (HLS) / 100 MHz (board) |
| Model Size | 391 KB (INT8 weights) |
| Latency | 2.3 s per image (HLS estimate @ 174 MHz) |

## Repository Structure

```
snn-fpga-shm/
├── 1_training/          PyTorch + SpikingJelly model training
├── 2_quantization/      INT8 post-training quantization & weight export
├── 3_hls/               Vitis HLS C++ design (SNN accelerator IP)
├── 4_deployment/        KV260 board deployment (bitstream, PYNQ driver, power)
├── requirements.txt     Python dependencies
└── .gitignore
```

## Requirements

### Software
- Python 3.10+
- PyTorch 1.13+ with CUDA support
- SpikingJelly
- Vitis HLS 2022.2
- Vivado 2022.2
- PYNQ v3.0 (on KV260 board)

### Hardware
- Xilinx Kria KV260 Vision AI Starter Kit
- Host PC with Vivado 2022.2 (for synthesis)
- SDNET2018 dataset (concrete crack images)

### Dataset
Download the SDNET2018 dataset and organize as:
```
SDNET2018/
├── D/  (Deck)
│   ├── CD/  (Cracked Deck)
│   └── UD/ (Uncracked Deck)
├── P/  (Pavement)
│   ├── CP/
│   └── UP/
└── W/  (Wall)
    ├── CW/
    └── UW/
```

## Quick Start

### Option A: Deploy Pre-Built Bitstream (Fastest)

Skip to `4_deployment/README.md` — load the provided bitstream on a KV260 and run inference immediately.

### Option B: Full Pipeline (Training to Deployment)

1. **Train the model**: See `1_training/README.md`
2. **Quantize to INT8**: See `2_quantization/README.md`
3. **Synthesize HLS IP**: See `3_hls/README.md`
4. **Deploy on KV260**: See `4_deployment/README.md`

## Architecture

The SNN implements a Spiking ResNet-18 (Variant D) with the following optimizations:

- **Multi-scale Membrane Fusion (MMF)**: Stem with 7x7 stride-4 convolution
- **Progressive Channel Scaling (PCS)**: Channel reduction (3 -> 16 -> 24 -> 48 -> 96)
- **Multi-Timestep Strategy (MTS)**: 10 timesteps with IF neurons
- **Channel Folding (F=4)**: Process 4 output channels per iteration, reducing BRAM by 4x

```
Input 256x256x3 (INT8)
  -> Stem: Conv7x7 s4, 3->16 ch     -> 64x64x16
  -> Stage1: 2x BasicBlock (16->16)  -> 64x64x16
  -> Stage2: BasicBlock s2 + BB      -> 32x32x24
  -> Stage3: BasicBlock s2 + BB      -> 16x16x48
  -> Stage4: BasicBlock s2 + BB      -> 8x8x96
  -> Global AvgPool -> FC(96->2) -> Argmax
```
