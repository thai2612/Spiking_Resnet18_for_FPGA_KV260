/**
 * @file snn_top_phase2E.h
 * @brief Top-level header for Phase 2E - Channel Folding
 */

#ifndef SNN_TOP_PHASE2E_H
#define SNN_TOP_PHASE2E_H

#include "snn_network_phase2E.h"

/**
 * @brief Top-level SNN inference function
 *
 * @param input Input image [INPUT_SIZE]
 * @param weights Network weights
 * @param output FC output logits [FC_OUT]
 * @return Predicted class (argmax)
 */
int snn_top(
    const act_t input[INPUT_SIZE],
    const NetworkWeights *weights,
    acc_t output[FC_OUT]
);

#endif // SNN_TOP_PHASE2E_H
