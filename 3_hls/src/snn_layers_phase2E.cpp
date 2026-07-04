/**
 * @file snn_layers_phase2E.cpp
 * @brief Layer implementations for Phase 2E - Channel Folding
 *
 * All folded functions process FOLD_SIZE output channels at a time.
 * This reduces intermediate buffer sizes while keeping membrane arrays full.
 */

#include "snn_layers_phase2E.h"

// ============================================================
// FOLDED CONVOLUTION
// ============================================================

void conv2d_folded(
    const act_t *input,
    const weight_t *weights,
    const bias_t *bias,
    membrane16_t *output,
    int IC, int IH, int IW,
    int OH, int OW,
    int K, int stride, int pad
) {
    // Process FOLD_SIZE output channels
    // Weight layout: [FOLD_SIZE][IC][K][K]

    CONV_FOLD_OC:
    for (int oc = 0; oc < FOLD_SIZE; oc++) {
        CONV_FOLD_OH:
        for (int oh = 0; oh < OH; oh++) {
            CONV_FOLD_OW:
            for (int ow = 0; ow < OW; ow++) {
                acc_t acc = (acc_t)bias[oc];

                CONV_FOLD_IC:
                for (int ic = 0; ic < IC; ic++) {
                    CONV_FOLD_KH:
                    for (int kh = 0; kh < K; kh++) {
                        CONV_FOLD_KW:
                        for (int kw = 0; kw < K; kw++) {
                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;

                            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                int in_idx = ic * IH * IW + ih * IW + iw;
                                int w_idx = oc * IC * K * K + ic * K * K + kh * K + kw;
                                acc += (acc_t)input[in_idx] * (acc_t)weights[w_idx];
                            }
                        }
                    }
                }

                int out_idx = oc * OH * OW + oh * OW + ow;
                output[out_idx] = SATURATE_INT16(acc >> 7);
            }
        }
    }
}

void conv1x1_folded(
    const act_t *input,
    const weight_t *weights,
    const bias_t *bias,
    membrane16_t *output,
    int IC, int IH, int IW,
    int OH, int OW,
    int stride
) {
    // Conv1x1 for FOLD_SIZE output channels
    // Weight layout: [FOLD_SIZE][IC]

    CONV1X1_FOLD_OC:
    for (int oc = 0; oc < FOLD_SIZE; oc++) {
        CONV1X1_FOLD_OH:
        for (int oh = 0; oh < OH; oh++) {
            CONV1X1_FOLD_OW:
            for (int ow = 0; ow < OW; ow++) {
                acc_t acc = (acc_t)bias[oc];

                int ih = oh * stride;
                int iw = ow * stride;

                CONV1X1_FOLD_IC:
                for (int ic = 0; ic < IC; ic++) {
                    int in_idx = ic * IH * IW + ih * IW + iw;
                    int w_idx = oc * IC + ic;
                    acc += (acc_t)input[in_idx] * (acc_t)weights[w_idx];
                }

                int out_idx = oc * OH * OW + oh * OW + ow;
                output[out_idx] = SATURATE_INT16(acc >> 7);
            }
        }
    }
}

// ============================================================
// FOLDED BATCHNORM
// ============================================================

void batchnorm_folded(
    const membrane16_t *input,
    const weight_t *scale,
    const bias_t *bias,
    membrane16_t *output,
    int H, int W
) {
    BN_FOLD_C:
    for (int c = 0; c < FOLD_SIZE; c++) {
        BN_FOLD_H:
        for (int h = 0; h < H; h++) {
            BN_FOLD_W:
            for (int w = 0; w < W; w++) {
                int idx = c * H * W + h * W + w;
                acc_t val = (acc_t)input[idx];
                acc_t s = (acc_t)scale[c];
                acc_t b = (acc_t)bias[c];
                acc_t y = ((val * s) >> 7) + b;
                output[idx] = SATURATE_INT16(y);
            }
        }
    }
}

// ============================================================
// FOLDED IF NEURON
// ============================================================

void if_neuron_folded(
    const membrane16_t *input,
    membrane_t *membrane,
    spike_t *output,
    int H, int W
) {
    // Process FOLD_SIZE channels
    // Membrane and output are at correct offset (passed by caller)

    IF_FOLD_C:
    for (int c = 0; c < FOLD_SIZE; c++) {
        IF_FOLD_H:
        for (int h = 0; h < H; h++) {
            IF_FOLD_W:
            for (int w = 0; w < W; w++) {
                #pragma HLS PIPELINE II=1

                int idx = c * H * W + h * W + w;

                // Scale input from INT16 to INT8 range
                int16_t scaled_input = (int16_t)(input[idx] >> MEMBRANE_SCALE_SHIFT);
                int8_t scaled_clipped = SATURATE_INT8(scaled_input);

                // Integrate
                int16_t new_v = (int16_t)membrane[idx] + (int16_t)scaled_clipped;
                membrane_t v = SATURATE_INT8(new_v);

                // Fire
                if (v >= IF_THRESHOLD) {
                    output[idx] = 1;
                    membrane[idx] = IF_RESET;
                } else {
                    output[idx] = 0;
                    membrane[idx] = v;
                }
            }
        }
    }
}

// ============================================================
// FOLDED UTILITIES
// ============================================================

void shortcut_identity_folded(
    const act_t *input,
    membrane16_t *output,
    int H, int W
) {
    // Scale INT8 input to INT16 for residual add
    SHORTCUT_FOLD_C:
    for (int c = 0; c < FOLD_SIZE; c++) {
        SHORTCUT_FOLD_H:
        for (int h = 0; h < H; h++) {
            SHORTCUT_FOLD_W:
            for (int w = 0; w < W; w++) {
                #pragma HLS PIPELINE II=1
                int idx = c * H * W + h * W + w;
                output[idx] = ((membrane16_t)input[idx]) << 7;
            }
        }
    }
}

void residual_add_folded(
    const membrane16_t *a,
    const membrane16_t *b,
    membrane16_t *output,
    int H, int W
) {
    RESADD_FOLD_C:
    for (int c = 0; c < FOLD_SIZE; c++) {
        RESADD_FOLD_H:
        for (int h = 0; h < H; h++) {
            RESADD_FOLD_W:
            for (int w = 0; w < W; w++) {
                #pragma HLS PIPELINE II=1
                int idx = c * H * W + h * W + w;
                acc_t sum = (acc_t)a[idx] + (acc_t)b[idx];
                output[idx] = SATURATE_INT16(sum);
            }
        }
    }
}

void spike_to_act_folded(
    const spike_t *spikes,
    act_t *activations,
    int H, int W
) {
    S2A_FOLD_C:
    for (int c = 0; c < FOLD_SIZE; c++) {
        S2A_FOLD_H:
        for (int h = 0; h < H; h++) {
            S2A_FOLD_W:
            for (int w = 0; w < W; w++) {
                #pragma HLS PIPELINE II=1
                int idx = c * H * W + h * W + w;
                activations[idx] = spikes[idx] ? 127 : 0;
            }
        }
    }
}

// ============================================================
// NON-FOLDED UTILITIES
// ============================================================

void reset_membrane_int8(membrane_t *membrane, int size) {
    RESET_MEM:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        membrane[i] = 0;
    }
}

void global_avg_pool(const spike_t *input, acc_t *output, int C, int H, int W) {
    GAP_C:
    for (int c = 0; c < C; c++) {
        acc_t sum = 0;
        GAP_H:
        for (int h = 0; h < H; h++) {
            GAP_W:
            for (int w = 0; w < W; w++) {
                int idx = c * H * W + h * W + w;
                sum += (acc_t)input[idx];
            }
        }
        output[c] = sum;
    }
}

void fc_layer(
    const acc_t *input,
    const weight_t *weights,
    const bias_t *bias,
    acc_t *output,
    int in_features,
    int out_features
) {
    FC_O_LOOP:
    for (int o = 0; o < out_features; o++) {
        acc_t acc = (acc_t)bias[o] << 7;

        FC_I_LOOP:
        for (int i = 0; i < in_features; i++) {
            int w_idx = o * in_features + i;
            acc += input[i] * (acc_t)weights[w_idx];
        }

        output[o] = acc >> 7;
    }
}
