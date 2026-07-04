# 3. HLS: Vitis HLS SNN Accelerator Design

C++ implementation of the channel-folding SNN accelerator for Vitis HLS 2022.2.

## Files

```
3_hls/
├── src/
│   ├── snn_config_phase2E.h        Architecture constants & data types
│   ├── snn_layers_phase2E.h/.cpp   Folded conv, BN, IF neuron primitives
│   ├── snn_network_phase2E.h/.cpp  Complete network dataflow (604 lines)
│   └── snn_top_phase2E.h/.cpp      Top-level AXI interface + argmax
├── tb/
│   └── tb_phase2E.cpp              C testbench (loads weights/images, checks golden)
├── scripts/
│   ├── run_csim.tcl                C simulation
│   ├── run_csynth.tcl              High-level synthesis
│   ├── run_cosim.tcl               Synthesis + RTL co-simulation
│   └── run_export.tcl              Export packaged IP
└── data/
    ├── network_weights.bin         INT8 weights (404 KB)
    ├── test_input_00.bin           Test image (256x256x3, INT8)
    ├── test_input_01.bin           Test image
    └── golden_output_00.txt        Golden reference logits
```

## Design Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `T_STEPS` | 10 | SNN timesteps |
| `FOLD_SIZE` | 4 | Output channels per fold iteration |
| `IF_THRESHOLD` | 64 | Neuron fire threshold |
| `MEMBRANE_SCALE_SHIFT` | 8 | INT16 to INT8 rescaling |
| Input | 256x256x3 | INT8 image |
| Output | 2 logits | Binary classification (crack/no-crack) |

### Data Types

| Type | Width | Usage |
|------|-------|-------|
| `weight_t` | int8 | Conv/FC weights |
| `act_t` | int8 | Activations |
| `bias_t` | int16 | BatchNorm bias |
| `membrane_t` | int8 | Membrane potential (all stages) |
| `membrane16_t` | int16 | Fold buffers (conv/bn/shortcut accumulation) |
| `acc_t` | int32 | Accumulators |

### Memory Binding (from run_csynth.tcl)

| Storage | Resource | Arrays |
|---------|----------|--------|
| Stem + Stage1 membrane | BRAM | 5 arrays |
| Stage2-4 membrane | **URAM** | 12 arrays |
| Fold buffers (conv/bn/shortcut/act) | BRAM | 4 arrays |
| Spike/activation arrays | BRAM | full-size |

## How to Run

### Prerequisites
- Vitis HLS 2022.2
- Target part: `xczu5ev-sfvc784-2-i` (Zynq UltraScale+ ZU5EV)
- Clock: 5 ns (200 MHz target, achieves ~174 MHz)

### C Simulation

```bash
cd scripts
vitis_hls -f run_csim.tcl
```

Runs the C testbench with test data from `../data/`. Expected output:
```
CSIM PASSED: test_input_00 -> Class 0 (expected 0)
```

### High-Level Synthesis

```bash
vitis_hls -f run_csynth.tcl
```

Generates RTL and synthesis reports (resource utilization, latency, timing).

### RTL Co-Simulation

```bash
vitis_hls -f run_cosim.tcl
```

Runs C/RTL co-simulation with XSIM (Verilog). Requires prior csynth.

### Export IP

```bash
vitis_hls -f run_export.tcl
```

Packages the design as a reusable IP core for Vivado block design integration.

## Synthesis Results (C-Synthesis)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| BRAM_18K | 245 | 288 | 85% |
| URAM | 44 | 64 | 68% |
| DSP | 55 | 1,248 | 4% |
| LUT | 86,012 | 117,120 | 73% |
| FF | 41,428 | 234,240 | 17% |

Latency: ~401M cycles (2.3 s at 174 MHz).

## Network Topology

```
Input 256x256x3 (INT8)
  -> Stem:   Conv7x7 s4,  3->16  ->  64x64x16
  -> Stage1: 2x BasicBlock, 16->16  ->  64x64x16
  -> Stage2: BasicBlock s2 + BB     ->  32x32x24
  -> Stage3: BasicBlock s2 + BB     ->  16x16x48
  -> Stage4: BasicBlock s2 + BB     ->  8x8x96
  -> GlobalAvgPool -> FC(96->2) -> Argmax
```

Each BasicBlock processes `FOLD_SIZE=4` output channels per iteration, reducing intermediate buffers from full channel width to 4x64x64 = 16,384 elements (vs 16x64x64 = 65,536 without folding).
