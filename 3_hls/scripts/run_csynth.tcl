# ============================================================
# SNN ResNet-18 Variant D - Channel Folding (FINAL DESIGN)
# Synthesis Script
# ============================================================
# Results:
#   BRAM: 245/288 (85%)  - PASSED
#   URAM: 44/64 (68%)    - PASSED
#   DSP:  55/1248 (4%)   - PASSED
#   LUT:  86012 (73%)    - PASSED
#   Timing: 5.745ns (~175 MHz)
# ============================================================

open_project -reset snn_project
set_top snn_top

# Source files
add_files ../src/snn_config_phase2E.h
add_files ../src/snn_layers_phase2E.h
add_files ../src/snn_layers_phase2E.cpp
add_files ../src/snn_network_phase2E.h
add_files ../src/snn_network_phase2E.cpp
add_files ../src/snn_top_phase2E.h
add_files ../src/snn_top_phase2E.cpp

open_solution -reset "solution1"
set_part {xczu5ev-sfvc784-2-i}
create_clock -period 5 -name default

# ============================================================
# MEMORY BINDING DIRECTIVES
# ============================================================

# Stage 2 membranes -> URAM (24576 elements each)
set_directive_bind_storage -type RAM_S2P -impl URAM "snn_network_inference" mem_s2b0_mid
set_directive_bind_storage -type RAM_S2P -impl URAM "snn_network_inference" mem_s2b0_out
set_directive_bind_storage -type RAM_S2P -impl URAM "snn_network_inference" mem_s2b1_mid
set_directive_bind_storage -type RAM_S2P -impl URAM "snn_network_inference" mem_s2b1_out

# Stage 3 membranes -> URAM (12288 elements each)
set_directive_bind_storage -type RAM_S2P -impl URAM "snn_network_inference" mem_s3b0_mid
set_directive_bind_storage -type RAM_S2P -impl URAM "snn_network_inference" mem_s3b0_out
set_directive_bind_storage -type RAM_S2P -impl URAM "snn_network_inference" mem_s3b1_mid
set_directive_bind_storage -type RAM_S2P -impl URAM "snn_network_inference" mem_s3b1_out

# Stage 4 membranes -> URAM (6144 elements each)
set_directive_bind_storage -type RAM_S2P -impl URAM "snn_network_inference" mem_s4b0_mid
set_directive_bind_storage -type RAM_S2P -impl URAM "snn_network_inference" mem_s4b0_out
set_directive_bind_storage -type RAM_S2P -impl URAM "snn_network_inference" mem_s4b1_mid
set_directive_bind_storage -type RAM_S2P -impl URAM "snn_network_inference" mem_s4b1_out

# Stem and Stage 1 -> BRAM (explicit)
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" mem_stem
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" mem_s1b0_mid
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" mem_s1b0_out
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" mem_s1b1_mid
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" mem_s1b1_out

# Folded buffers -> BRAM (small, 16384 elements)
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" conv_fold
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" bn_fold
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" shortcut_fold
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" act_fold

# Full spike/act arrays -> BRAM
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" stem_spikes
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" stem_act
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" s1_spikes
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" s1_act
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" s2_spikes
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" s2_act
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" s3_spikes
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" s3_act
set_directive_bind_storage -type RAM_S2P -impl BRAM "snn_network_inference" s4_spikes

# Run synthesis
csynth_design

puts "============================================================"
puts "CSYNTH Complete - Channel Folding (FINAL DESIGN)"
puts "============================================================"
puts "Check report: snn_project/solution1/syn/report/"
puts "============================================================"
