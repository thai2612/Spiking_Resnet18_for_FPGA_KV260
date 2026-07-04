# 2. Quantization: INT8 Post-Training Quantization & Weight Export

Converts the trained FP32 model to INT8 fixed-point and exports the C header files consumed by the HLS design.

## Files

```
2_quantization/
├── scripts/
│   ├── quantize_model.py         PTQ + weight export to C arrays
│   ├── export_golden_ref.py      Generate golden reference vectors
│   └── compare_quantization.py   Compare FP32 vs INT8 vs INT16
└── weights/
    └── format.md                 Weight format documentation
```

## Pipeline

```
Trained checkpoint (.pth)
        |
        v
compare_quantization.py -----> quantization_comparison.txt (decide INT8/INT16)
        |
        v
quantize_model.py -----------> weights.h + weights.c (INT8 C arrays)
        |                      quantization_params.txt (per-layer scales)
        v
export_golden_ref.py --------> golden_ref/ (per-layer outputs for HLS verification)
```

## Setup

The scripts import model definitions from `1_training/`. Ensure the folder structure is:

```
snn-fpga-shm/
├── 1_training/
│   └── custom_spiking_resnet_hls.py
└── 2_quantization/
    └── scripts/
        └── *.py
```

Install dependencies:

```bash
pip install -r ../requirements.txt
```

## Step 1: Compare Quantization Precision

```bash
cd scripts
python compare_quantization.py \
    --checkpoint /path/to/best_spiking_resnet_hlsD.pth \
    --variant D \
    --data-dir /path/to/SDNET2018
```

This evaluates FP32, INT8, and INT16 precision and recommends a quantization scheme.

### Example Output

```
=== Quantization Comparison ===
Precision   Accuracy    Memory    Avg Error
FP32        86.35%      1565 KB   ---
INT8        86.34%      391 KB    0.002
INT16       86.35%      783 KB    0.0002

Recommendation: INT8 (accuracy drop < 0.1%, 4x memory reduction)
```

## Step 2: Export INT8 Weights

```bash
python quantize_model.py \
    --checkpoint /path/to/best_spiking_resnet_hlsD.pth \
    --variant D \
    --method ptq \
    --bits 8 \
    --data-dir /path/to/SDNET2018
```

### Output

```
weights/
├── weights.h      C header with extern declarations + scale macros
├── weights.c      ~2.4 MB of int8_t arrays (packed weights)
└── format.md      Layer-by-layer format documentation

quantization_output/
├── weight_statistics.txt     Per-layer min/max/mean/std
└── quantization_params.txt   Per-layer scale factors
```

### Quantization Method

- **Scheme**: Per-tensor symmetric INT8 (Q7)
- **Scale**: `scale = abs_max / 127`
- **Calibration**: 50 batches of validation data for activation ranges
- **Right-shift**: Accumulator >> 7 in fixed-point MAC operations

## Step 3: Generate Golden References

```bash
python export_golden_ref.py \
    --checkpoint /path/to/best_spiking_resnet_hlsD.pth \
    --variant D \
    --num-samples 100 \
    --data-dir /path/to/SDNET2018
```

Generates per-layer output tensors for bit-exact verification of the HLS C model.

## Weight Format

See `weights/format.md` for the complete specification. Key points:

- All weights are `int8_t` (INT8, symmetric, range [-127, 127])
- Each layer has a `*_SCALE` macro (fixed-point scale factor)
- Layer order in `weights.c` matches the `NetworkWeights` struct in the HLS source
- The packed binary `network_weights.bin` (404 KB) used by the HLS testbench and board deployment is the flat binary concatenation of all weight arrays in `NetworkWeights` struct order. A pre-built copy is provided in `3_hls/data/` and `4_deployment/pynq/`.
