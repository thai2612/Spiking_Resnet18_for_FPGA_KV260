# ============================================================
# SNN ResNet-18 Variant D - Channel Folding (FINAL DESIGN)
# Export IP Script
# ============================================================
# Prerequisites: Must run run_csynth.tcl first
# ============================================================

open_project snn_project
open_solution "solution1"

# Export IP for Vivado integration
export_design -format ip_catalog -description "SNN ResNet-18 Variant D Accelerator" -vendor "custom" -version "1.0"

puts "============================================================"
puts "Export Complete - Channel Folding (FINAL DESIGN)"
puts "============================================================"
puts "IP exported to: snn_project/solution1/impl/ip/"
puts "============================================================"
