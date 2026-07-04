/**
 * @file snn_top_phase2E.cpp
 * @brief Top-level implementation for Phase 2E - Channel Folding
 */

#include "snn_top_phase2E.h"

int snn_top(
    const act_t input[INPUT_SIZE],
    const NetworkWeights *weights,
    acc_t output[FC_OUT]
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Run network inference
    snn_network_inference(input, weights, output);

    // Argmax
    int pred = 0;
    acc_t max_val = output[0];

    ARGMAX:
    for (int i = 1; i < FC_OUT; i++) {
        #pragma HLS PIPELINE II=1
        if (output[i] > max_val) {
            max_val = output[i];
            pred = i;
        }
    }

    return pred;
}
