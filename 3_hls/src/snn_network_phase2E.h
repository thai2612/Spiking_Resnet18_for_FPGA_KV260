/**
 * @file snn_network_phase2E.h
 * @brief Network header for Phase 2E - Channel Folding
 */

#ifndef SNN_NETWORK_PHASE2E_H
#define SNN_NETWORK_PHASE2E_H

#include "snn_config_phase2E.h"
#include "snn_layers_phase2E.h"

// ============================================================
// Weight Structure
// ============================================================

struct NetworkWeights {
    // Stem
    weight_t stem_conv_w[STEM_CONV_W_SIZE];
    bias_t   stem_conv_b[STEM_OUT_C];
    weight_t stem_bn_s[STEM_OUT_C];
    bias_t   stem_bn_b[STEM_OUT_C];

    // Stage 1 Block 0
    weight_t s1b0_conv1_w[S1B0_CONV1_W_SIZE];
    bias_t   s1b0_conv1_b[16];
    weight_t s1b0_bn1_s[16];
    bias_t   s1b0_bn1_b[16];
    weight_t s1b0_conv2_w[S1B0_CONV2_W_SIZE];
    bias_t   s1b0_conv2_b[16];
    weight_t s1b0_bn2_s[16];
    bias_t   s1b0_bn2_b[16];

    // Stage 1 Block 1
    weight_t s1b1_conv1_w[S1B1_CONV1_W_SIZE];
    bias_t   s1b1_conv1_b[16];
    weight_t s1b1_bn1_s[16];
    bias_t   s1b1_bn1_b[16];
    weight_t s1b1_conv2_w[S1B1_CONV2_W_SIZE];
    bias_t   s1b1_conv2_b[16];
    weight_t s1b1_bn2_s[16];
    bias_t   s1b1_bn2_b[16];

    // Stage 2 Block 0 (with shortcut)
    weight_t s2b0_conv1_w[S2B0_CONV1_W_SIZE];
    bias_t   s2b0_conv1_b[24];
    weight_t s2b0_bn1_s[24];
    bias_t   s2b0_bn1_b[24];
    weight_t s2b0_conv2_w[S2B0_CONV2_W_SIZE];
    bias_t   s2b0_conv2_b[24];
    weight_t s2b0_bn2_s[24];
    bias_t   s2b0_bn2_b[24];
    weight_t s2b0_short_w[S2B0_SHORT_W_SIZE];
    bias_t   s2b0_short_b[24];
    weight_t s2b0_short_bn_s[24];
    bias_t   s2b0_short_bn_b[24];

    // Stage 2 Block 1
    weight_t s2b1_conv1_w[S2B1_CONV1_W_SIZE];
    bias_t   s2b1_conv1_b[24];
    weight_t s2b1_bn1_s[24];
    bias_t   s2b1_bn1_b[24];
    weight_t s2b1_conv2_w[S2B1_CONV2_W_SIZE];
    bias_t   s2b1_conv2_b[24];
    weight_t s2b1_bn2_s[24];
    bias_t   s2b1_bn2_b[24];

    // Stage 3 Block 0 (with shortcut)
    weight_t s3b0_conv1_w[S3B0_CONV1_W_SIZE];
    bias_t   s3b0_conv1_b[48];
    weight_t s3b0_bn1_s[48];
    bias_t   s3b0_bn1_b[48];
    weight_t s3b0_conv2_w[S3B0_CONV2_W_SIZE];
    bias_t   s3b0_conv2_b[48];
    weight_t s3b0_bn2_s[48];
    bias_t   s3b0_bn2_b[48];
    weight_t s3b0_short_w[S3B0_SHORT_W_SIZE];
    bias_t   s3b0_short_b[48];
    weight_t s3b0_short_bn_s[48];
    bias_t   s3b0_short_bn_b[48];

    // Stage 3 Block 1
    weight_t s3b1_conv1_w[S3B1_CONV1_W_SIZE];
    bias_t   s3b1_conv1_b[48];
    weight_t s3b1_bn1_s[48];
    bias_t   s3b1_bn1_b[48];
    weight_t s3b1_conv2_w[S3B1_CONV2_W_SIZE];
    bias_t   s3b1_conv2_b[48];
    weight_t s3b1_bn2_s[48];
    bias_t   s3b1_bn2_b[48];

    // Stage 4 Block 0 (with shortcut)
    weight_t s4b0_conv1_w[S4B0_CONV1_W_SIZE];
    bias_t   s4b0_conv1_b[96];
    weight_t s4b0_bn1_s[96];
    bias_t   s4b0_bn1_b[96];
    weight_t s4b0_conv2_w[S4B0_CONV2_W_SIZE];
    bias_t   s4b0_conv2_b[96];
    weight_t s4b0_bn2_s[96];
    bias_t   s4b0_bn2_b[96];
    weight_t s4b0_short_w[S4B0_SHORT_W_SIZE];
    bias_t   s4b0_short_b[96];
    weight_t s4b0_short_bn_s[96];
    bias_t   s4b0_short_bn_b[96];

    // Stage 4 Block 1
    weight_t s4b1_conv1_w[S4B1_CONV1_W_SIZE];
    bias_t   s4b1_conv1_b[96];
    weight_t s4b1_bn1_s[96];
    bias_t   s4b1_bn1_b[96];
    weight_t s4b1_conv2_w[S4B1_CONV2_W_SIZE];
    bias_t   s4b1_conv2_b[96];
    weight_t s4b1_bn2_s[96];
    bias_t   s4b1_bn2_b[96];

    // FC
    weight_t fc_w[FC_W_SIZE];
    bias_t   fc_b[FC_OUT];
};

// ============================================================
// Network Inference
// ============================================================

void snn_network_inference(
    const act_t input[INPUT_SIZE],
    const NetworkWeights *weights,
    acc_t output[FC_OUT]
);

#endif // SNN_NETWORK_PHASE2E_H
