/**
 * @file snn_layers_phase2E.h
 * @brief Layer declarations for Phase 2E - Channel Folding
 */

#ifndef SNN_LAYERS_PHASE2E_H
#define SNN_LAYERS_PHASE2E_H

#include "snn_config_phase2E.h"

// ============================================================
// FOLDED CONVOLUTION
// ============================================================

/**
 * @brief Conv2d processing FOLD_SIZE output channels
 *
 * @param input Full input tensor [IC][IH][IW]
 * @param weights Weights for this fold [FOLD_SIZE][IC][K][K]
 * @param bias Bias for this fold [FOLD_SIZE]
 * @param output Output buffer [FOLD_SIZE][OH][OW]
 * @param IC Input channels (full)
 * @param IH, IW Input spatial dimensions
 * @param OH, OW Output spatial dimensions
 * @param K Kernel size
 * @param stride Convolution stride
 * @param pad Padding
 */
void conv2d_folded(
    const act_t *input,
    const weight_t *weights,
    const bias_t *bias,
    membrane16_t *output,
    int IC, int IH, int IW,
    int OH, int OW,
    int K, int stride, int pad
);

/**
 * @brief Conv1x1 for shortcut, processing FOLD_SIZE output channels
 */
void conv1x1_folded(
    const act_t *input,
    const weight_t *weights,
    const bias_t *bias,
    membrane16_t *output,
    int IC, int IH, int IW,
    int OH, int OW,
    int stride
);

// ============================================================
// FOLDED BATCHNORM
// ============================================================

/**
 * @brief Batchnorm for FOLD_SIZE channels
 *
 * @param input Input buffer [FOLD_SIZE][H][W]
 * @param scale Scale params for this fold [FOLD_SIZE]
 * @param bias Bias params for this fold [FOLD_SIZE]
 * @param output Output buffer [FOLD_SIZE][H][W]
 * @param H, W Spatial dimensions
 */
void batchnorm_folded(
    const membrane16_t *input,
    const weight_t *scale,
    const bias_t *bias,
    membrane16_t *output,
    int H, int W
);

// ============================================================
// FOLDED IF NEURON
// ============================================================

/**
 * @brief IF neuron for FOLD_SIZE channels, writes directly to output
 *
 * @param input Input buffer [FOLD_SIZE][H][W]
 * @param membrane Membrane state at offset (full array)
 * @param output Spike output at offset (full array)
 * @param H, W Spatial dimensions
 */
void if_neuron_folded(
    const membrane16_t *input,
    membrane_t *membrane,
    spike_t *output,
    int H, int W
);

// ============================================================
// FOLDED UTILITIES
// ============================================================

/**
 * @brief Compute shortcut (identity) for FOLD_SIZE channels
 */
void shortcut_identity_folded(
    const act_t *input,
    membrane16_t *output,
    int H, int W
);

/**
 * @brief Residual add for FOLD_SIZE channels
 */
void residual_add_folded(
    const membrane16_t *a,
    const membrane16_t *b,
    membrane16_t *output,
    int H, int W
);

/**
 * @brief Convert spikes to activations for FOLD_SIZE channels
 */
void spike_to_act_folded(
    const spike_t *spikes,
    act_t *activations,
    int H, int W
);

// ============================================================
// NON-FOLDED UTILITIES (for final stages)
// ============================================================

void reset_membrane_int8(membrane_t *membrane, int size);
void global_avg_pool(const spike_t *input, acc_t *output, int C, int H, int W);
void fc_layer(const acc_t *input, const weight_t *weights, const bias_t *bias,
              acc_t *output, int in_features, int out_features);

#endif // SNN_LAYERS_PHASE2E_H
