# ============================================================
# SNN ResNet-18 Variant D - Channel Folding (FINAL DESIGN)
# C Simulation Script
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

# Testbench
add_files -tb ../tb/tb_phase2E.cpp
add_files -tb ../data

open_solution -reset "solution1"
set_part {xczu5ev-sfvc784-2-i}
create_clock -period 5 -name default

# Run C simulation
csim_design -argv "../data"

puts "============================================================"
puts "CSIM Complete - Channel Folding (FINAL DESIGN)"
puts "============================================================"
puts "If PASSED, run CSYNTH:"
puts "  vitis_hls -f run_csynth.tcl"
puts "============================================================"
