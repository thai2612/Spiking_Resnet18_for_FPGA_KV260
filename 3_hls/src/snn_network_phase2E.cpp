/**
 * @file snn_network_phase2E.cpp
 * @brief Network implementation for Phase 2E - Channel Folding
 *
 * CHANNEL FOLDING STRATEGY:
 * - Process FOLD_SIZE=4 output channels per iteration
 * - Intermediate buffers (conv_fold, bn_fold, shortcut_fold) are small
 * - Membrane arrays remain full size (IF persistence)
 * - Spike outputs written directly to full arrays
 *
 * BUFFER REDUCTION:
 * - conv_temp: 65536 -> 16384 (4x)
 * - bn_temp: 65536 -> 16384 (4x)
 * - spike_temp: eliminated (write direct)
 * - shared_shortcut: 65536 -> 16384 (4x)
 */

#include "snn_network_phase2E.h"

// ============================================================
// FOLDED INTERMEDIATE BUFFERS (small)
// ============================================================

static membrane16_t conv_fold[FOLD_BUFFER_MAX];      // 16384
static membrane16_t bn_fold[FOLD_BUFFER_MAX];        // 16384
static membrane16_t shortcut_fold[FOLD_BUFFER_MAX];  // 16384

// Activation buffer for conv2 input (after spike_to_act)
// Size: max(FOLD_SIZE * H * W) across all stages = 16384
static act_t act_fold[FOLD_BUFFER_MAX];

// ============================================================
// MEMBRANE ARRAYS - Full size (NOT folded)
// ============================================================

static membrane_t mem_stem[MEMBRANE_STEM_SIZE];

static membrane_t mem_s1b0_mid[STAGE1_FM_SIZE];
static membrane_t mem_s1b0_out[STAGE1_FM_SIZE];
static membrane_t mem_s1b1_mid[STAGE1_FM_SIZE];
static membrane_t mem_s1b1_out[STAGE1_FM_SIZE];

static membrane_t mem_s2b0_mid[STAGE2_FM_SIZE];
static membrane_t mem_s2b0_out[STAGE2_FM_SIZE];
static membrane_t mem_s2b1_mid[STAGE2_FM_SIZE];
static membrane_t mem_s2b1_out[STAGE2_FM_SIZE];

static membrane_t mem_s3b0_mid[STAGE3_FM_SIZE];
static membrane_t mem_s3b0_out[STAGE3_FM_SIZE];
static membrane_t mem_s3b1_mid[STAGE3_FM_SIZE];
static membrane_t mem_s3b1_out[STAGE3_FM_SIZE];

static membrane_t mem_s4b0_mid[STAGE4_FM_SIZE];
static membrane_t mem_s4b0_out[STAGE4_FM_SIZE];
static membrane_t mem_s4b1_mid[STAGE4_FM_SIZE];
static membrane_t mem_s4b1_out[STAGE4_FM_SIZE];

// ============================================================
// SPIKE/ACTIVATION ARRAYS - Full size (output of each layer)
// ============================================================

static spike_t stem_spikes[STEM_FM_SIZE];
static act_t stem_act[STEM_FM_SIZE];

static spike_t s1_spikes[STAGE1_FM_SIZE];
static act_t s1_act[STAGE1_FM_SIZE];

static spike_t s2_spikes[STAGE2_FM_SIZE];
static act_t s2_act[STAGE2_FM_SIZE];

static spike_t s3_spikes[STAGE3_FM_SIZE];
static act_t s3_act[STAGE3_FM_SIZE];

static spike_t s4_spikes[STAGE4_FM_SIZE];

// FC accumulator
static acc_t fc_acc[FC_IN];

// ============================================================
// PROCESS STEM WITH FOLDING
// ============================================================

static void process_stem_folded(
    const act_t *input,
    const NetworkWeights *weights,
    membrane_t *membrane,
    spike_t *output
) {
    STEM_FOLD:
    for (int fold = 0; fold < STEM_NUM_FOLDS; fold++) {
        #pragma HLS LOOP_TRIPCOUNT min=4 max=4

        int oc_start = fold * FOLD_SIZE;
        int spatial_size = STEM_OUT_H * STEM_OUT_W;
        int offset = oc_start * spatial_size;

        // Weight offset: [oc_start][IC][K][K]
        int w_offset = oc_start * INPUT_C * STEM_K * STEM_K;

        // Conv 7x7 for FOLD_SIZE channels
        conv2d_folded(
            input,
            &weights->stem_conv_w[w_offset],
            &weights->stem_conv_b[oc_start],
            conv_fold,
            INPUT_C, INPUT_H, INPUT_W,
            STEM_OUT_H, STEM_OUT_W,
            STEM_K, STEM_S, STEM_P
        );

        // BatchNorm
        batchnorm_folded(
            conv_fold,
            &weights->stem_bn_s[oc_start],
            &weights->stem_bn_b[oc_start],
            bn_fold,
            STEM_OUT_H, STEM_OUT_W
        );

        // IF neuron - write directly to output at offset
        if_neuron_folded(
            bn_fold,
            &membrane[offset],
            &output[offset],
            STEM_OUT_H, STEM_OUT_W
        );
    }
}

// ============================================================
// PROCESS BASICBLOCK WITH FOLDING (no downsample)
// ============================================================

static void process_basicblock_folded(
    const act_t *input,
    const weight_t *conv1_w, const bias_t *conv1_b,
    const weight_t *bn1_s, const bias_t *bn1_b,
    const weight_t *conv2_w, const bias_t *conv2_b,
    const weight_t *bn2_s, const bias_t *bn2_b,
    membrane_t *membrane_mid,
    membrane_t *membrane_out,
    spike_t *mid_spikes,      // Temporary for conv2 input
    spike_t *output,
    int C, int H, int W
) {
    int spatial_size = H * W;
    int num_folds = C / FOLD_SIZE;

    // ============ PATH 1: Conv1 -> BN1 -> IF1 ============
    BLOCK_FOLD1:
    for (int fold = 0; fold < num_folds; fold++) {
        #pragma HLS LOOP_TRIPCOUNT min=4 max=24

        int oc_start = fold * FOLD_SIZE;
        int offset = oc_start * spatial_size;

        // Conv1 weights: [oc_start][C][3][3]
        int w1_offset = oc_start * C * 9;

        conv2d_folded(
            input,
            &conv1_w[w1_offset],
            &conv1_b[oc_start],
            conv_fold,
            C, H, W,
            H, W,
            BLOCK_K, 1, BLOCK_P
        );

        batchnorm_folded(
            conv_fold,
            &bn1_s[oc_start],
            &bn1_b[oc_start],
            bn_fold,
            H, W
        );

        // IF1 - write to mid_spikes at offset
        if_neuron_folded(
            bn_fold,
            &membrane_mid[offset],
            &mid_spikes[offset],
            H, W
        );
    }

    // Convert all mid_spikes to activations for conv2 input
    // (Full conversion, not folded, because conv2 needs all IC)
    SPIKE_TO_ACT:
    for (int i = 0; i < C * H * W; i++) {
        #pragma HLS PIPELINE II=1
        s1_act[i] = mid_spikes[i] ? 127 : 0;  // Reuse s1_act as temp
    }

    // ============ PATH 2: Conv2 -> BN2 -> Residual -> IF2 ============
    BLOCK_FOLD2:
    for (int fold = 0; fold < num_folds; fold++) {
        #pragma HLS LOOP_TRIPCOUNT min=4 max=24

        int oc_start = fold * FOLD_SIZE;
        int offset = oc_start * spatial_size;

        // Conv2 weights: [oc_start][C][3][3]
        int w2_offset = oc_start * C * 9;

        conv2d_folded(
            s1_act,  // Use converted activations
            &conv2_w[w2_offset],
            &conv2_b[oc_start],
            conv_fold,
            C, H, W,
            H, W,
            BLOCK_K, 1, BLOCK_P
        );

        batchnorm_folded(
            conv_fold,
            &bn2_s[oc_start],
            &bn2_b[oc_start],
            bn_fold,
            H, W
        );

        // Identity shortcut for this fold
        shortcut_identity_folded(
            &input[offset],
            shortcut_fold,
            H, W
        );

        // Residual add
        residual_add_folded(bn_fold, shortcut_fold, conv_fold, H, W);

        // IF2 - write to output at offset
        if_neuron_folded(
            conv_fold,
            &membrane_out[offset],
            &output[offset],
            H, W
        );
    }
}

// ============================================================
// PROCESS BASICBLOCK WITH DOWNSAMPLE AND FOLDING
// ============================================================

static void process_basicblock_down_folded(
    const act_t *input,
    const weight_t *conv1_w, const bias_t *conv1_b,
    const weight_t *bn1_s, const bias_t *bn1_b,
    const weight_t *conv2_w, const bias_t *conv2_b,
    const weight_t *bn2_s, const bias_t *bn2_b,
    const weight_t *short_w, const bias_t *short_b,
    const weight_t *short_bn_s, const bias_t *short_bn_b,
    membrane_t *membrane_mid,
    membrane_t *membrane_out,
    spike_t *mid_spikes,
    act_t *mid_act,
    spike_t *output,
    int IC, int IH, int IW,
    int OC, int OH, int OW
) {
    int out_spatial = OH * OW;
    int num_folds = OC / FOLD_SIZE;

    // ============ PATH 1: Conv1(s=2) -> BN1 -> IF1 ============
    DOWN_FOLD1:
    for (int fold = 0; fold < num_folds; fold++) {
        #pragma HLS LOOP_TRIPCOUNT min=6 max=24

        int oc_start = fold * FOLD_SIZE;
        int offset = oc_start * out_spatial;

        // Conv1 weights: [oc_start][IC][3][3]
        int w1_offset = oc_start * IC * 9;

        conv2d_folded(
            input,
            &conv1_w[w1_offset],
            &conv1_b[oc_start],
            conv_fold,
            IC, IH, IW,
            OH, OW,
            BLOCK_K, 2, BLOCK_P  // stride=2
        );

        batchnorm_folded(
            conv_fold,
            &bn1_s[oc_start],
            &bn1_b[oc_start],
            bn_fold,
            OH, OW
        );

        if_neuron_folded(
            bn_fold,
            &membrane_mid[offset],
            &mid_spikes[offset],
            OH, OW
        );
    }

    // Convert mid_spikes to activations
    int mid_size = OC * OH * OW;
    SPIKE_TO_ACT_DOWN:
    for (int i = 0; i < mid_size; i++) {
        #pragma HLS PIPELINE II=1
        mid_act[i] = mid_spikes[i] ? 127 : 0;
    }

    // ============ PATH 2: Conv2 -> BN2 ============
    // ============ SHORTCUT: Conv1x1(s=2) -> BN ============
    // ============ Residual Add -> IF2 ============
    DOWN_FOLD2:
    for (int fold = 0; fold < num_folds; fold++) {
        #pragma HLS LOOP_TRIPCOUNT min=6 max=24

        int oc_start = fold * FOLD_SIZE;
        int offset = oc_start * out_spatial;

        // Conv2 weights: [oc_start][OC][3][3]
        int w2_offset = oc_start * OC * 9;

        conv2d_folded(
            mid_act,
            &conv2_w[w2_offset],
            &conv2_b[oc_start],
            conv_fold,
            OC, OH, OW,
            OH, OW,
            BLOCK_K, 1, BLOCK_P
        );

        batchnorm_folded(
            conv_fold,
            &bn2_s[oc_start],
            &bn2_b[oc_start],
            bn_fold,
            OH, OW
        );

        // Shortcut: Conv1x1 stride 2
        // short_w layout: [OC][IC] -> offset [oc_start][IC]
        int sw_offset = oc_start * IC;

        conv1x1_folded(
            input,
            &short_w[sw_offset],
            &short_b[oc_start],
            shortcut_fold,
            IC, IH, IW,
            OH, OW,
            2  // stride=2
        );

        // Shortcut BN
        batchnorm_folded(
            shortcut_fold,
            &short_bn_s[oc_start],
            &short_bn_b[oc_start],
            shortcut_fold,  // in-place OK for BN
            OH, OW
        );

        // Residual add
        residual_add_folded(bn_fold, shortcut_fold, conv_fold, OH, OW);

        // IF2
        if_neuron_folded(
            conv_fold,
            &membrane_out[offset],
            &output[offset],
            OH, OW
        );
    }
}

// ============================================================
// RESET ALL MEMBRANES
// ============================================================

static void reset_all_membranes() {
    reset_membrane_int8(mem_stem, MEMBRANE_STEM_SIZE);

    reset_membrane_int8(mem_s1b0_mid, STAGE1_FM_SIZE);
    reset_membrane_int8(mem_s1b0_out, STAGE1_FM_SIZE);
    reset_membrane_int8(mem_s1b1_mid, STAGE1_FM_SIZE);
    reset_membrane_int8(mem_s1b1_out, STAGE1_FM_SIZE);

    reset_membrane_int8(mem_s2b0_mid, STAGE2_FM_SIZE);
    reset_membrane_int8(mem_s2b0_out, STAGE2_FM_SIZE);
    reset_membrane_int8(mem_s2b1_mid, STAGE2_FM_SIZE);
    reset_membrane_int8(mem_s2b1_out, STAGE2_FM_SIZE);

    reset_membrane_int8(mem_s3b0_mid, STAGE3_FM_SIZE);
    reset_membrane_int8(mem_s3b0_out, STAGE3_FM_SIZE);
    reset_membrane_int8(mem_s3b1_mid, STAGE3_FM_SIZE);
    reset_membrane_int8(mem_s3b1_out, STAGE3_FM_SIZE);

    reset_membrane_int8(mem_s4b0_mid, STAGE4_FM_SIZE);
    reset_membrane_int8(mem_s4b0_out, STAGE4_FM_SIZE);
    reset_membrane_int8(mem_s4b1_mid, STAGE4_FM_SIZE);
    reset_membrane_int8(mem_s4b1_out, STAGE4_FM_SIZE);

    FC_RESET:
    for (int i = 0; i < FC_IN; i++) {
        #pragma HLS PIPELINE II=1
        fc_acc[i] = 0;
    }
}

// ============================================================
// FULL NETWORK INFERENCE
// ============================================================

void snn_network_inference(
    const act_t input[INPUT_SIZE],
    const NetworkWeights *weights,
    acc_t output[FC_OUT]
) {
    acc_t pooled[FC_IN];

    reset_all_membranes();

    TIMESTEP_LOOP:
    for (int t = 0; t < T_STEPS; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=10 max=10

        // ========== STEM (folded) ==========
        process_stem_folded(input, weights, mem_stem, stem_spikes);

        // Convert stem spikes to activations (full)
        STEM_S2A:
        for (int i = 0; i < STEM_FM_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            stem_act[i] = stem_spikes[i] ? 127 : 0;
        }

        // ========== STAGE 1 (folded) ==========
        process_basicblock_folded(
            stem_act,
            weights->s1b0_conv1_w, weights->s1b0_conv1_b,
            weights->s1b0_bn1_s, weights->s1b0_bn1_b,
            weights->s1b0_conv2_w, weights->s1b0_conv2_b,
            weights->s1b0_bn2_s, weights->s1b0_bn2_b,
            mem_s1b0_mid, mem_s1b0_out,
            s1_spikes, s1_spikes,
            STAGE1_OUT_C, STAGE1_H, STAGE1_W
        );

        S1B0_S2A:
        for (int i = 0; i < STAGE1_FM_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            s1_act[i] = s1_spikes[i] ? 127 : 0;
        }

        process_basicblock_folded(
            s1_act,
            weights->s1b1_conv1_w, weights->s1b1_conv1_b,
            weights->s1b1_bn1_s, weights->s1b1_bn1_b,
            weights->s1b1_conv2_w, weights->s1b1_conv2_b,
            weights->s1b1_bn2_s, weights->s1b1_bn2_b,
            mem_s1b1_mid, mem_s1b1_out,
            s1_spikes, s1_spikes,
            STAGE1_OUT_C, STAGE1_H, STAGE1_W
        );

        S1B1_S2A:
        for (int i = 0; i < STAGE1_FM_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            s1_act[i] = s1_spikes[i] ? 127 : 0;
        }

        // ========== STAGE 2 (folded with downsample) ==========
        process_basicblock_down_folded(
            s1_act,
            weights->s2b0_conv1_w, weights->s2b0_conv1_b,
            weights->s2b0_bn1_s, weights->s2b0_bn1_b,
            weights->s2b0_conv2_w, weights->s2b0_conv2_b,
            weights->s2b0_bn2_s, weights->s2b0_bn2_b,
            weights->s2b0_short_w, weights->s2b0_short_b,
            weights->s2b0_short_bn_s, weights->s2b0_short_bn_b,
            mem_s2b0_mid, mem_s2b0_out,
            s2_spikes, s2_act, s2_spikes,
            STAGE1_OUT_C, STAGE1_H, STAGE1_W,
            STAGE2_OUT_C, STAGE2_H, STAGE2_W
        );

        S2B0_S2A:
        for (int i = 0; i < STAGE2_FM_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            s2_act[i] = s2_spikes[i] ? 127 : 0;
        }

        process_basicblock_folded(
            s2_act,
            weights->s2b1_conv1_w, weights->s2b1_conv1_b,
            weights->s2b1_bn1_s, weights->s2b1_bn1_b,
            weights->s2b1_conv2_w, weights->s2b1_conv2_b,
            weights->s2b1_bn2_s, weights->s2b1_bn2_b,
            mem_s2b1_mid, mem_s2b1_out,
            s2_spikes, s2_spikes,
            STAGE2_OUT_C, STAGE2_H, STAGE2_W
        );

        S2B1_S2A:
        for (int i = 0; i < STAGE2_FM_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            s2_act[i] = s2_spikes[i] ? 127 : 0;
        }

        // ========== STAGE 3 (folded with downsample) ==========
        process_basicblock_down_folded(
            s2_act,
            weights->s3b0_conv1_w, weights->s3b0_conv1_b,
            weights->s3b0_bn1_s, weights->s3b0_bn1_b,
            weights->s3b0_conv2_w, weights->s3b0_conv2_b,
            weights->s3b0_bn2_s, weights->s3b0_bn2_b,
            weights->s3b0_short_w, weights->s3b0_short_b,
            weights->s3b0_short_bn_s, weights->s3b0_short_bn_b,
            mem_s3b0_mid, mem_s3b0_out,
            s3_spikes, s3_act, s3_spikes,
            STAGE2_OUT_C, STAGE2_H, STAGE2_W,
            STAGE3_OUT_C, STAGE3_H, STAGE3_W
        );

        S3B0_S2A:
        for (int i = 0; i < STAGE3_FM_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            s3_act[i] = s3_spikes[i] ? 127 : 0;
        }

        process_basicblock_folded(
            s3_act,
            weights->s3b1_conv1_w, weights->s3b1_conv1_b,
            weights->s3b1_bn1_s, weights->s3b1_bn1_b,
            weights->s3b1_conv2_w, weights->s3b1_conv2_b,
            weights->s3b1_bn2_s, weights->s3b1_bn2_b,
            mem_s3b1_mid, mem_s3b1_out,
            s3_spikes, s3_spikes,
            STAGE3_OUT_C, STAGE3_H, STAGE3_W
        );

        S3B1_S2A:
        for (int i = 0; i < STAGE3_FM_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            s3_act[i] = s3_spikes[i] ? 127 : 0;
        }

        // ========== STAGE 4 (folded with downsample) ==========
        static act_t s4_act[STAGE4_FM_SIZE];

        process_basicblock_down_folded(
            s3_act,
            weights->s4b0_conv1_w, weights->s4b0_conv1_b,
            weights->s4b0_bn1_s, weights->s4b0_bn1_b,
            weights->s4b0_conv2_w, weights->s4b0_conv2_b,
            weights->s4b0_bn2_s, weights->s4b0_bn2_b,
            weights->s4b0_short_w, weights->s4b0_short_b,
            weights->s4b0_short_bn_s, weights->s4b0_short_bn_b,
            mem_s4b0_mid, mem_s4b0_out,
            s4_spikes, s4_act, s4_spikes,
            STAGE3_OUT_C, STAGE3_H, STAGE3_W,
            STAGE4_OUT_C, STAGE4_H, STAGE4_W
        );

        S4B0_S2A:
        for (int i = 0; i < STAGE4_FM_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            s4_act[i] = s4_spikes[i] ? 127 : 0;
        }

        process_basicblock_folded(
            s4_act,
            weights->s4b1_conv1_w, weights->s4b1_conv1_b,
            weights->s4b1_bn1_s, weights->s4b1_bn1_b,
            weights->s4b1_conv2_w, weights->s4b1_conv2_b,
            weights->s4b1_bn2_s, weights->s4b1_bn2_b,
            mem_s4b1_mid, mem_s4b1_out,
            s4_spikes, s4_spikes,
            STAGE4_OUT_C, STAGE4_H, STAGE4_W
        );

        // ========== POOLING ==========
        global_avg_pool(s4_spikes, pooled, STAGE4_OUT_C, STAGE4_H, STAGE4_W);

        FC_ACC:
        for (int i = 0; i < FC_IN; i++) {
            #pragma HLS PIPELINE II=1
            fc_acc[i] += pooled[i];
        }
    }

    // ========== FINAL FC ==========
    int spatial_size = STAGE4_H * STAGE4_W;
    FC_NORM:
    for (int i = 0; i < FC_IN; i++) {
        #pragma HLS PIPELINE II=1
        fc_acc[i] = fc_acc[i] / spatial_size;
    }

    fc_layer(fc_acc, weights->fc_w, weights->fc_b, output, FC_IN, FC_OUT);
}
