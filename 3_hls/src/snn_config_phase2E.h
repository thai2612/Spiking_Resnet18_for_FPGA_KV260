/**
 * @file snn_config_phase2E.h
 * @brief Configuration for Phase 2E - Channel Folding
 *
 * CHANNEL FOLDING STRATEGY:
 * - Process output channels in groups of FOLD_SIZE
 * - Reduce intermediate buffer sizes by fold factor
 * - Eliminate spike_temp (write directly to output)
 * - Keep membrane arrays full size (IF persistence)
 *
 * EXPECTED SAVINGS:
 * - conv_temp: 65536 -> 16384 (4x reduction)
 * - bn_temp: 65536 -> 16384 (4x reduction)
 * - spike_temp: eliminated
 * - shared_shortcut: 65536 -> 16384 (4x reduction)
 * - Total: ~158 BRAM saved
 */

#ifndef SNN_CONFIG_PHASE2E_H
#define SNN_CONFIG_PHASE2E_H

#include <stdint.h>

// ============================================================
// CHANNEL FOLDING CONFIGURATION
// ============================================================

#define FOLD_SIZE 4  // Process 4 output channels at a time

// ============================================================
// DATA TYPES
// ============================================================

typedef int8_t   weight_t;
typedef int8_t   act_t;
typedef int16_t  bias_t;
typedef int8_t   membrane_t;    // INT8 membrane
typedef int16_t  membrane16_t;  // INT16 intermediate
typedef int32_t  acc_t;
typedef uint8_t  spike_t;

// ============================================================
// NETWORK ARCHITECTURE (Variant D)
// ============================================================

#define T_STEPS 10

// Input
#define INPUT_H 256
#define INPUT_W 256
#define INPUT_C 3
#define INPUT_SIZE (INPUT_C * INPUT_H * INPUT_W)

// Stem: Conv7x7(s=4) -> 64x64x16
#define STEM_OUT_H 64
#define STEM_OUT_W 64
#define STEM_OUT_C 16
#define STEM_K 7
#define STEM_S 4
#define STEM_P 3
#define STEM_FM_SIZE (STEM_OUT_C * STEM_OUT_H * STEM_OUT_W)

// Stage 1: 64x64x16
#define STAGE1_H 64
#define STAGE1_W 64
#define STAGE1_IN_C 16
#define STAGE1_OUT_C 16
#define STAGE1_FM_SIZE (STAGE1_OUT_C * STAGE1_H * STAGE1_W)

// Stage 2: 32x32x24
#define STAGE2_H 32
#define STAGE2_W 32
#define STAGE2_IN_C 16
#define STAGE2_OUT_C 24
#define STAGE2_FM_SIZE (STAGE2_OUT_C * STAGE2_H * STAGE2_W)

// Stage 3: 16x16x48
#define STAGE3_H 16
#define STAGE3_W 16
#define STAGE3_IN_C 24
#define STAGE3_OUT_C 48
#define STAGE3_FM_SIZE (STAGE3_OUT_C * STAGE3_H * STAGE3_W)

// Stage 4: 8x8x96
#define STAGE4_H 8
#define STAGE4_W 8
#define STAGE4_IN_C 48
#define STAGE4_OUT_C 96
#define STAGE4_FM_SIZE (STAGE4_OUT_C * STAGE4_H * STAGE4_W)

// FC
#define FC_IN 96
#define FC_OUT 2

// BasicBlock kernel
#define BLOCK_K 3
#define BLOCK_P 1

// ============================================================
// FOLD BUFFER SIZES
// ============================================================

// Maximum fold buffer = FOLD_SIZE * max(H*W)
// Stem/S1: 4 * 64 * 64 = 16384
// S2: 4 * 32 * 32 = 4096
// S3: 4 * 16 * 16 = 1024
// S4: 4 * 8 * 8 = 256

#define FOLD_BUFFER_MAX (FOLD_SIZE * STEM_OUT_H * STEM_OUT_W)  // 16384

// Number of folds per stage
#define STEM_NUM_FOLDS (STEM_OUT_C / FOLD_SIZE)    // 16/4 = 4
#define S1_NUM_FOLDS   (STAGE1_OUT_C / FOLD_SIZE)  // 16/4 = 4
#define S2_NUM_FOLDS   (STAGE2_OUT_C / FOLD_SIZE)  // 24/4 = 6
#define S3_NUM_FOLDS   (STAGE3_OUT_C / FOLD_SIZE)  // 48/4 = 12
#define S4_NUM_FOLDS   (STAGE4_OUT_C / FOLD_SIZE)  // 96/4 = 24

// ============================================================
// MEMBRANE SIZES (Full - NOT folded)
// ============================================================

#define MEMBRANE_STEM_SIZE STEM_FM_SIZE
#define MEMBRANE_STAGE1_SIZE STAGE1_FM_SIZE
#define MEMBRANE_STAGE2_SIZE STAGE2_FM_SIZE
#define MEMBRANE_STAGE3_SIZE STAGE3_FM_SIZE
#define MEMBRANE_STAGE4_SIZE STAGE4_FM_SIZE

#define MAX_FM_SIZE STAGE1_FM_SIZE  // 65536

// ============================================================
// IF NEURON PARAMETERS
// ============================================================

#define IF_THRESHOLD 64
#define IF_RESET 0
#define MEMBRANE_SCALE_SHIFT 8

// ============================================================
// WEIGHT ARRAY SIZES
// ============================================================

#define STEM_CONV_W_SIZE (STEM_OUT_C * INPUT_C * STEM_K * STEM_K)

#define S1B0_CONV1_W_SIZE (16 * 16 * 3 * 3)
#define S1B0_CONV2_W_SIZE (16 * 16 * 3 * 3)
#define S1B1_CONV1_W_SIZE (16 * 16 * 3 * 3)
#define S1B1_CONV2_W_SIZE (16 * 16 * 3 * 3)

#define S2B0_CONV1_W_SIZE (24 * 16 * 3 * 3)
#define S2B0_CONV2_W_SIZE (24 * 24 * 3 * 3)
#define S2B0_SHORT_W_SIZE (24 * 16 * 1 * 1)
#define S2B1_CONV1_W_SIZE (24 * 24 * 3 * 3)
#define S2B1_CONV2_W_SIZE (24 * 24 * 3 * 3)

#define S3B0_CONV1_W_SIZE (48 * 24 * 3 * 3)
#define S3B0_CONV2_W_SIZE (48 * 48 * 3 * 3)
#define S3B0_SHORT_W_SIZE (48 * 24 * 1 * 1)
#define S3B1_CONV1_W_SIZE (48 * 48 * 3 * 3)
#define S3B1_CONV2_W_SIZE (48 * 48 * 3 * 3)

#define S4B0_CONV1_W_SIZE (96 * 48 * 3 * 3)
#define S4B0_CONV2_W_SIZE (96 * 96 * 3 * 3)
#define S4B0_SHORT_W_SIZE (96 * 48 * 1 * 1)
#define S4B1_CONV1_W_SIZE (96 * 96 * 3 * 3)
#define S4B1_CONV2_W_SIZE (96 * 96 * 3 * 3)

#define FC_W_SIZE (FC_OUT * FC_IN)

// ============================================================
// HELPER MACROS
// ============================================================

#define SATURATE_INT8(x) \
    (((x) > 127) ? 127 : (((x) < -128) ? -128 : (x)))

#define SATURATE_INT16(x) \
    (((x) > 32767) ? 32767 : (((x) < -32768) ? -32768 : (x)))

#endif // SNN_CONFIG_PHASE2E_H
