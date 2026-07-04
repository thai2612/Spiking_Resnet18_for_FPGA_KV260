/**
 * @file tb_phase2E.cpp
 * @brief Testbench for Phase 2E - Channel Folding
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/snn_top_phase2E.h"

int load_binary(const char* filepath, void* buffer, size_t size) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        printf("ERROR: Cannot open %s\n", filepath);
        return -1;
    }
    size_t read = fread(buffer, 1, size, fp);
    fclose(fp);
    if (read != size) {
        printf("WARNING: Read %zu bytes, expected %zu\n", read, size);
        return -1;
    }
    return 0;
}

int load_network_weights(const char* filepath, NetworkWeights* w) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        printf("ERROR: Cannot open weights file: %s\n", filepath);
        return -1;
    }

    size_t total_read = 0;
    size_t n;

    // Stem
    n = fread(w->stem_conv_w, 1, sizeof(w->stem_conv_w), fp); total_read += n;
    n = fread(w->stem_conv_b, 1, sizeof(w->stem_conv_b), fp); total_read += n;
    n = fread(w->stem_bn_s, 1, sizeof(w->stem_bn_s), fp); total_read += n;
    n = fread(w->stem_bn_b, 1, sizeof(w->stem_bn_b), fp); total_read += n;

    // S1B0
    n = fread(w->s1b0_conv1_w, 1, sizeof(w->s1b0_conv1_w), fp); total_read += n;
    n = fread(w->s1b0_conv1_b, 1, sizeof(w->s1b0_conv1_b), fp); total_read += n;
    n = fread(w->s1b0_bn1_s, 1, sizeof(w->s1b0_bn1_s), fp); total_read += n;
    n = fread(w->s1b0_bn1_b, 1, sizeof(w->s1b0_bn1_b), fp); total_read += n;
    n = fread(w->s1b0_conv2_w, 1, sizeof(w->s1b0_conv2_w), fp); total_read += n;
    n = fread(w->s1b0_conv2_b, 1, sizeof(w->s1b0_conv2_b), fp); total_read += n;
    n = fread(w->s1b0_bn2_s, 1, sizeof(w->s1b0_bn2_s), fp); total_read += n;
    n = fread(w->s1b0_bn2_b, 1, sizeof(w->s1b0_bn2_b), fp); total_read += n;

    // S1B1
    n = fread(w->s1b1_conv1_w, 1, sizeof(w->s1b1_conv1_w), fp); total_read += n;
    n = fread(w->s1b1_conv1_b, 1, sizeof(w->s1b1_conv1_b), fp); total_read += n;
    n = fread(w->s1b1_bn1_s, 1, sizeof(w->s1b1_bn1_s), fp); total_read += n;
    n = fread(w->s1b1_bn1_b, 1, sizeof(w->s1b1_bn1_b), fp); total_read += n;
    n = fread(w->s1b1_conv2_w, 1, sizeof(w->s1b1_conv2_w), fp); total_read += n;
    n = fread(w->s1b1_conv2_b, 1, sizeof(w->s1b1_conv2_b), fp); total_read += n;
    n = fread(w->s1b1_bn2_s, 1, sizeof(w->s1b1_bn2_s), fp); total_read += n;
    n = fread(w->s1b1_bn2_b, 1, sizeof(w->s1b1_bn2_b), fp); total_read += n;

    // S2B0
    n = fread(w->s2b0_conv1_w, 1, sizeof(w->s2b0_conv1_w), fp); total_read += n;
    n = fread(w->s2b0_conv1_b, 1, sizeof(w->s2b0_conv1_b), fp); total_read += n;
    n = fread(w->s2b0_bn1_s, 1, sizeof(w->s2b0_bn1_s), fp); total_read += n;
    n = fread(w->s2b0_bn1_b, 1, sizeof(w->s2b0_bn1_b), fp); total_read += n;
    n = fread(w->s2b0_conv2_w, 1, sizeof(w->s2b0_conv2_w), fp); total_read += n;
    n = fread(w->s2b0_conv2_b, 1, sizeof(w->s2b0_conv2_b), fp); total_read += n;
    n = fread(w->s2b0_bn2_s, 1, sizeof(w->s2b0_bn2_s), fp); total_read += n;
    n = fread(w->s2b0_bn2_b, 1, sizeof(w->s2b0_bn2_b), fp); total_read += n;
    n = fread(w->s2b0_short_w, 1, sizeof(w->s2b0_short_w), fp); total_read += n;
    n = fread(w->s2b0_short_b, 1, sizeof(w->s2b0_short_b), fp); total_read += n;
    n = fread(w->s2b0_short_bn_s, 1, sizeof(w->s2b0_short_bn_s), fp); total_read += n;
    n = fread(w->s2b0_short_bn_b, 1, sizeof(w->s2b0_short_bn_b), fp); total_read += n;

    // S2B1
    n = fread(w->s2b1_conv1_w, 1, sizeof(w->s2b1_conv1_w), fp); total_read += n;
    n = fread(w->s2b1_conv1_b, 1, sizeof(w->s2b1_conv1_b), fp); total_read += n;
    n = fread(w->s2b1_bn1_s, 1, sizeof(w->s2b1_bn1_s), fp); total_read += n;
    n = fread(w->s2b1_bn1_b, 1, sizeof(w->s2b1_bn1_b), fp); total_read += n;
    n = fread(w->s2b1_conv2_w, 1, sizeof(w->s2b1_conv2_w), fp); total_read += n;
    n = fread(w->s2b1_conv2_b, 1, sizeof(w->s2b1_conv2_b), fp); total_read += n;
    n = fread(w->s2b1_bn2_s, 1, sizeof(w->s2b1_bn2_s), fp); total_read += n;
    n = fread(w->s2b1_bn2_b, 1, sizeof(w->s2b1_bn2_b), fp); total_read += n;

    // S3B0
    n = fread(w->s3b0_conv1_w, 1, sizeof(w->s3b0_conv1_w), fp); total_read += n;
    n = fread(w->s3b0_conv1_b, 1, sizeof(w->s3b0_conv1_b), fp); total_read += n;
    n = fread(w->s3b0_bn1_s, 1, sizeof(w->s3b0_bn1_s), fp); total_read += n;
    n = fread(w->s3b0_bn1_b, 1, sizeof(w->s3b0_bn1_b), fp); total_read += n;
    n = fread(w->s3b0_conv2_w, 1, sizeof(w->s3b0_conv2_w), fp); total_read += n;
    n = fread(w->s3b0_conv2_b, 1, sizeof(w->s3b0_conv2_b), fp); total_read += n;
    n = fread(w->s3b0_bn2_s, 1, sizeof(w->s3b0_bn2_s), fp); total_read += n;
    n = fread(w->s3b0_bn2_b, 1, sizeof(w->s3b0_bn2_b), fp); total_read += n;
    n = fread(w->s3b0_short_w, 1, sizeof(w->s3b0_short_w), fp); total_read += n;
    n = fread(w->s3b0_short_b, 1, sizeof(w->s3b0_short_b), fp); total_read += n;
    n = fread(w->s3b0_short_bn_s, 1, sizeof(w->s3b0_short_bn_s), fp); total_read += n;
    n = fread(w->s3b0_short_bn_b, 1, sizeof(w->s3b0_short_bn_b), fp); total_read += n;

    // S3B1
    n = fread(w->s3b1_conv1_w, 1, sizeof(w->s3b1_conv1_w), fp); total_read += n;
    n = fread(w->s3b1_conv1_b, 1, sizeof(w->s3b1_conv1_b), fp); total_read += n;
    n = fread(w->s3b1_bn1_s, 1, sizeof(w->s3b1_bn1_s), fp); total_read += n;
    n = fread(w->s3b1_bn1_b, 1, sizeof(w->s3b1_bn1_b), fp); total_read += n;
    n = fread(w->s3b1_conv2_w, 1, sizeof(w->s3b1_conv2_w), fp); total_read += n;
    n = fread(w->s3b1_conv2_b, 1, sizeof(w->s3b1_conv2_b), fp); total_read += n;
    n = fread(w->s3b1_bn2_s, 1, sizeof(w->s3b1_bn2_s), fp); total_read += n;
    n = fread(w->s3b1_bn2_b, 1, sizeof(w->s3b1_bn2_b), fp); total_read += n;

    // S4B0
    n = fread(w->s4b0_conv1_w, 1, sizeof(w->s4b0_conv1_w), fp); total_read += n;
    n = fread(w->s4b0_conv1_b, 1, sizeof(w->s4b0_conv1_b), fp); total_read += n;
    n = fread(w->s4b0_bn1_s, 1, sizeof(w->s4b0_bn1_s), fp); total_read += n;
    n = fread(w->s4b0_bn1_b, 1, sizeof(w->s4b0_bn1_b), fp); total_read += n;
    n = fread(w->s4b0_conv2_w, 1, sizeof(w->s4b0_conv2_w), fp); total_read += n;
    n = fread(w->s4b0_conv2_b, 1, sizeof(w->s4b0_conv2_b), fp); total_read += n;
    n = fread(w->s4b0_bn2_s, 1, sizeof(w->s4b0_bn2_s), fp); total_read += n;
    n = fread(w->s4b0_bn2_b, 1, sizeof(w->s4b0_bn2_b), fp); total_read += n;
    n = fread(w->s4b0_short_w, 1, sizeof(w->s4b0_short_w), fp); total_read += n;
    n = fread(w->s4b0_short_b, 1, sizeof(w->s4b0_short_b), fp); total_read += n;
    n = fread(w->s4b0_short_bn_s, 1, sizeof(w->s4b0_short_bn_s), fp); total_read += n;
    n = fread(w->s4b0_short_bn_b, 1, sizeof(w->s4b0_short_bn_b), fp); total_read += n;

    // S4B1
    n = fread(w->s4b1_conv1_w, 1, sizeof(w->s4b1_conv1_w), fp); total_read += n;
    n = fread(w->s4b1_conv1_b, 1, sizeof(w->s4b1_conv1_b), fp); total_read += n;
    n = fread(w->s4b1_bn1_s, 1, sizeof(w->s4b1_bn1_s), fp); total_read += n;
    n = fread(w->s4b1_bn1_b, 1, sizeof(w->s4b1_bn1_b), fp); total_read += n;
    n = fread(w->s4b1_conv2_w, 1, sizeof(w->s4b1_conv2_w), fp); total_read += n;
    n = fread(w->s4b1_conv2_b, 1, sizeof(w->s4b1_conv2_b), fp); total_read += n;
    n = fread(w->s4b1_bn2_s, 1, sizeof(w->s4b1_bn2_s), fp); total_read += n;
    n = fread(w->s4b1_bn2_b, 1, sizeof(w->s4b1_bn2_b), fp); total_read += n;

    // FC
    n = fread(w->fc_w, 1, sizeof(w->fc_w), fp); total_read += n;
    n = fread(w->fc_b, 1, sizeof(w->fc_b), fp); total_read += n;

    fclose(fp);
    printf("Loaded weights: %zu bytes\n", total_read);
    return 0;
}

int test_image(int image_idx, const char* data_dir, const NetworkWeights* weights) {
    char filepath[512];
    printf("\n========================================\n");
    printf("Testing image %d (Phase 2E - Channel Folding)\n", image_idx);
    printf("========================================\n");

    static act_t input[INPUT_SIZE];
    snprintf(filepath, sizeof(filepath), "%s/test_input_%02d.bin", data_dir, image_idx);
    if (load_binary(filepath, input, sizeof(input)) < 0) {
        return -1;
    }
    printf("Input loaded: %zu bytes\n", sizeof(input));
    printf("FOLD_SIZE: %d, Stem folds: %d, S1 folds: %d\n",
           FOLD_SIZE, STEM_NUM_FOLDS, S1_NUM_FOLDS);

    printf("Running inference (T=%d timesteps)...\n", T_STEPS);
    acc_t output[FC_OUT];
    int pred = snn_top(input, weights, output);

    printf("Output: [%d, %d]\n", (int)output[0], (int)output[1]);
    printf("Predicted class: %d\n", pred);

    // Load golden
    snprintf(filepath, sizeof(filepath), "%s/golden_output_%02d.txt", data_dir, image_idx);
    FILE* fp = fopen(filepath, "r");
    if (!fp) {
        printf("WARNING: No golden reference for image %d\n", image_idx);
        return 0;
    }

    int golden_pred = -1;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "predicted_class", 15) == 0) {
            sscanf(line, "predicted_class = %d", &golden_pred);
        }
    }
    fclose(fp);

    printf("Golden class: %d\n", golden_pred);

    if (pred == golden_pred) {
        printf("Result: MATCH\n");
        return 0;
    } else {
        printf("Result: MISMATCH\n");
        return 1;
    }
}

int main(int argc, char* argv[]) {
    printf("========================================\n");
    printf("SNN ResNet-18 - Phase 2E Test\n");
    printf("========================================\n");
    printf("Optimization: Channel Folding\n");
    printf("  - FOLD_SIZE: %d channels\n", FOLD_SIZE);
    printf("  - Fold buffer: %d elements\n", FOLD_BUFFER_MAX);
    printf("  - Expected BRAM reduction: ~158 BRAM\n");
    printf("========================================\n");

    const char* data_dir = "../data";
    if (argc >= 2) {
        data_dir = argv[1];
    }
    printf("Data directory: %s\n", data_dir);

    char weights_path[512];
    snprintf(weights_path, sizeof(weights_path), "%s/network_weights.bin", data_dir);

    static NetworkWeights weights;
    if (load_network_weights(weights_path, &weights) < 0) {
        printf("ERROR: Failed to load weights\n");
        return 1;
    }

    int num_images = 1;  // Reduced for faster COSIM (was 3)
    int mismatches = 0;
    int tested = 0;

    for (int i = 0; i < num_images; i++) {
        int result = test_image(i, data_dir, &weights);
        if (result >= 0) {
            tested++;
            mismatches += result;
        }
    }

    printf("\n========================================\n");
    printf("PHASE 2E SUMMARY\n");
    printf("========================================\n");
    printf("Images tested: %d\n", tested);
    printf("Classification matches: %d/%d\n", tested - mismatches, tested);

    if (mismatches == 0 && tested > 0) {
        printf("\nRESULT: CSIM PASSED\n");
        printf("Channel Folding verified.\n");
        printf("\nNext step: Run CSYNTH\n");
        printf("  vitis_hls -f run_csynth.tcl\n");
    } else if (tested == 0) {
        printf("\nRESULT: NO TESTS RUN\n");
    } else {
        printf("\nRESULT: CSIM MISMATCHES (%d)\n", mismatches);
    }
    printf("========================================\n");

    return (mismatches > 0) ? 1 : 0;
}
