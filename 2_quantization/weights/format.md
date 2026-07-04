# Quantized Weight Format

## Overview

- **Quantization**: 8-bit symmetric
- **Format**: Q7 (1 sign bit, 7 fractional bits for INT8)
- **Data Type**: `int8_t` (range: -128 to 127)

## Conversion

```c
// To convert INT8 back to float:
float fp_value = (float)int8_value * SCALE;

// For MAC operations:
int32_t acc = 0;
for (int i = 0; i < N; i++) {
    acc += (int32_t)input[i] * (int32_t)weight[i];
}
// Scale result: output_scale = input_scale * weight_scale
```

## Layer Scales

| Layer | Scale | Range |
|-------|-------|-------|
| model.stem.0.weight | 3.763518e-03 | [-127, 79] |
| model.stem.1.weight | 9.385038e-03 | [51, 127] |
| model.stages.0.0.conv1.weight | 5.037708e-03 | [-101, 127] |
| model.stages.0.0.bn1.weight | 7.554961e-03 | [55, 127] |
| model.stages.0.0.conv2.weight | 4.559098e-03 | [-75, 127] |
| model.stages.0.0.bn2.weight | 8.063794e-03 | [22, 127] |
| model.stages.0.1.conv1.weight | 4.906853e-03 | [-127, 103] |
| model.stages.0.1.bn1.weight | 8.606578e-03 | [0, 127] |
| model.stages.0.1.conv2.weight | 4.035864e-03 | [-127, 95] |
| model.stages.0.1.bn2.weight | 6.912324e-03 | [24, 127] |
| model.stages.1.0.conv1.weight | 3.415895e-03 | [-79, 127] |
| model.stages.1.0.bn1.weight | 6.243057e-03 | [0, 127] |
| model.stages.1.0.conv2.weight | 3.449196e-03 | [-127, 121] |
| model.stages.1.0.bn2.weight | 5.865844e-03 | [0, 127] |
| model.stages.1.0.downsample.0.weight | 4.612522e-03 | [-127, 116] |
| model.stages.1.0.downsample.1.weight | 6.841096e-03 | [0, 127] |
| model.stages.1.1.conv1.weight | 2.991785e-03 | [-89, 127] |
| model.stages.1.1.bn1.weight | 5.675933e-03 | [0, 127] |
| model.stages.1.1.conv2.weight | 2.404917e-03 | [-118, 127] |
| model.stages.1.1.bn2.weight | 4.560686e-03 | [0, 127] |
| model.stages.2.0.conv1.weight | 2.590729e-03 | [-127, 103] |
| model.stages.2.0.bn1.weight | 3.418615e-03 | [0, 127] |
| model.stages.2.0.conv2.weight | 2.472328e-03 | [-118, 127] |
| model.stages.2.0.bn2.weight | 2.950670e-03 | [-10, 127] |
| model.stages.2.0.downsample.0.weight | 3.437365e-03 | [-127, 112] |
| model.stages.2.0.downsample.1.weight | 6.152296e-03 | [-2, 127] |
| model.stages.2.1.conv1.weight | 2.645070e-03 | [-123, 127] |
| model.stages.2.1.bn1.weight | 3.953527e-03 | [0, 127] |
| model.stages.2.1.conv2.weight | 2.623669e-03 | [-103, 127] |
| model.stages.2.1.bn2.weight | 4.976608e-03 | [-29, 127] |
| model.stages.3.0.conv1.weight | 2.805198e-03 | [-121, 127] |
| model.stages.3.0.bn1.weight | 3.292293e-03 | [0, 127] |
| model.stages.3.0.conv2.weight | 2.905677e-03 | [-92, 127] |
| model.stages.3.0.bn2.weight | 3.787198e-03 | [-74, 127] |
| model.stages.3.0.downsample.0.weight | 2.513042e-03 | [-127, 126] |
| model.stages.3.0.downsample.1.weight | 3.943971e-03 | [-17, 127] |
| model.stages.3.1.conv1.weight | 2.902489e-03 | [-84, 127] |
| model.stages.3.1.bn1.weight | 4.003200e-03 | [0, 127] |
| model.stages.3.1.conv2.weight | 1.524251e-03 | [-116, 127] |
| model.stages.3.1.bn2.weight | 5.041501e-03 | [-67, 127] |
| model.fc.weight | 5.169686e-03 | [-127, 126] |
