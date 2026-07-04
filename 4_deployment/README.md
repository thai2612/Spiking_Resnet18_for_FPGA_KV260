# 4. Deployment: KV260 Board Deployment

Pre-built bitstream, PYNQ drivers, and power measurement tools for the Xilinx Kria KV260 platform.

## Files

```
4_deployment/
├── bitstream/
│   ├── snn_kv260.bit          FPGA bitstream (5.8 MB)
│   └── snn_kv260.hwh          PYNQ hardware handoff (0.3 MB)
├── pynq/
│   ├── snn_driver_v3.py       PYNQ driver (MMIO register access)
│   ├── benchmark.py           Benchmark + accuracy test script
│   └── network_weights.bin    INT8 weights (404 KB)
├── test_data/
│   ├── test_input_00.bin      Test image (256x256x3, INT8)
│   ├── test_input_01.bin      Test image
│   ├── golden_output_00.txt   Golden reference logits
│   └── golden_output_01.txt   Golden reference logits
├── vivado/
│   ├── build_kv260_snn.tcl    Vivado build script
│   ├── snn_bd_kv260.tcl       Block design (Zynq PS + SmartConnect + HLS IP)
│   └── constraints/
│       └── kv260_snn.xdc      Physical constraints
└── power/
    └── measure_power.ipynb    Board-level power measurement (INA260)
```

## Quick Start: Run Inference on KV260

### Prerequisites
- KV260 running PYNQ v3.0 image
- SSH access to the board (default: `xilinx@192.168.1.106`, password: `xilinx`)

### Step 1: Transfer Files

```bash
# Create directory on the board
ssh xilinx@<KV260_IP> "mkdir -p ~/snn"

# Copy bitstream and hardware handoff
scp bitstream/snn_kv260.bit bitstream/snn_kv260.hwh xilinx@<KV260_IP>:~/snn/

# Copy PYNQ driver and weights
scp pynq/snn_driver_v3.py pynq/benchmark.py pynq/network_weights.bin xilinx@<KV260_IP>:~/snn/

# Copy test data and golden references
scp test_data/test_input_*.bin test_data/golden_output_*.txt xilinx@<KV260_IP>:~/snn/

# Copy power measurement notebook
scp power/measure_power.ipynb xilinx@<KV260_IP>:~/snn/
```

### Step 2: Run Inference

```bash
ssh xilinx@<KV260_IP>
cd ~/snn
source /usr/local/share/pynq-venv/bin/activate
python3 snn_driver_v3.py snn_kv260.bit network_weights.bin test_input_00.bin
```

### Expected Output

```
Image shape: (256, 256, 3), dtype: int8

=== RESULTS ===
Logits: [25030 -32803]
Predicted class: 0
Class meaning: No crack
```

### Step 3: Run Benchmark

```bash
python3 benchmark.py
```

Compares FPGA predictions against golden reference vectors and reports accuracy.

## Driver Details (`snn_driver_v3.py`)

### Register Map

| Address | Register | Purpose |
|---------|----------|---------|
| `0xA0000000 + 0x00` | AP_CTRL | bit0=ap_start, bit1=ap_done |
| `0xA0000000 + 0x10` | AP_RETURN | Predicted class (argmax) |
| `0xA0010000 + 0x10` | input_r | Input buffer address (64-bit) |
| `0xA0010000 + 0x1C` | weights | Weights buffer address (64-bit) |
| `0xA0010000 + 0x28` | output_r | Output buffer address (64-bit) |

### Inference Flow

1. Load bitstream via `Overlay(bitstream)`
2. Allocate contiguous memory buffers (CMA) via `pynq.allocate`
3. Write input image + weights to CMA buffers
4. Write buffer physical addresses to `s_axi_control_r` registers
5. Write `0x01` to AP_CTRL to start the accelerator
6. Poll AP_CTRL until `ap_done` (120s timeout)
7. Read AP_RETURN for predicted class
8. Read output logits from CMA buffer

## Power Measurement (`measure_power.ipynb`)

Run directly on the KV260 via Jupyter:

```bash
cd ~/snn
jupyter notebook measure_power.ipynb
```

Measures dynamic power using the onboard INA260 sensor:
- **Idle power**: 3.722 W (50 samples over 10s)
- **Active power**: 4.043 W (during continuous inference)
- **Dynamic power**: 0.321 W (Active - Idle)

Sensor: `/sys/class/hwmon/hwmon2/power1_input` (micro-watts)

## Rebuild Bitstream (Optional)

To regenerate the bitstream from the HLS IP:

1. Run HLS synthesis in `3_hls/scripts/` to produce the packaged IP
2. Update the `ip_dirs` path in `vivado/build_kv260_snn.tcl` to point to the exported IP location
3. Run the build:

```bash
cd vivado
vivado -mode batch -source build_kv260_snn.tcl
```

This requires Vivado 2022.2 with KV260 board files installed.

## Hardware Architecture

```
Zynq UltraScale+ PS
  |
  +-- M_AXI_HPM0_FPD --> SmartConnect --> HLS s_axi_control (0xA0000000)
  |                                   --> HLS s_axi_control_r (0xA0010000)
  |
  +-- S_AXI_HP0 <-- SmartConnect <-- HLS m_axi_gmem0 (input/output)
  +-- S_AXI_HP1 <-- SmartConnect <-- HLS m_axi_gmem1 (weights)

PL Clock: 100 MHz
```

No DMA — HLS masters read/write DDR directly via AXI HP ports.
