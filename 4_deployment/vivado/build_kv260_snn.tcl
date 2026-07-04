# =============================================================
# Vivado Build Script for SNN ResNet-18 on KRIA KV260
# =============================================================
# FPGA Part: xck26-sfvc784-2LV-c
# Board Part: xilinx.com:kv260_som:part0:1.4
# =============================================================

set project_name "snn_kv260"
set bd_name "snn_bd"
set part "xck26-sfvc784-2LV-c"

# HLS IP location
set ip_dirs "../hls/snn_accel/scripts/snn_project/solution1/impl/ip"

puts "============================================================="
puts " SNN ResNet-18 Accelerator Build for KRIA KV260"
puts " FPGA Part: $part"
puts "============================================================="

# =============================================================
# Create Project
# =============================================================
create_project $project_name ./$project_name -part $part -force

# Try to set board part (requires board files installed)
catch {set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]}

# =============================================================
# Add HLS IP Repository
# =============================================================
if {[file exists $ip_dirs]} {
    set_property ip_repo_paths [file normalize $ip_dirs] [current_project]
    update_ip_catalog -rebuild
    puts "INFO: Added IP repository: $ip_dirs"
} else {
    puts "ERROR: HLS IP not found at $ip_dirs"
    puts "       Please run HLS CSynth + Export first:"
    puts "       cd ../hls/snn_accel/scripts"
    puts "       vitis_hls -f run_csynth_kv260.tcl"
    puts "       vitis_hls -f run_export_kv260.tcl"
    exit 1
}

# =============================================================
# Add Constraints
# =============================================================
if {[file exists "constraints/kv260_snn.xdc"]} {
    add_files -fileset constrs_1 -norecurse constraints/kv260_snn.xdc
    puts "INFO: Added constraints file"
}

# =============================================================
# Create Block Design
# =============================================================
puts "INFO: Creating block design..."
source snn_bd_kv260.tcl
create_snn_bd $bd_name
puts "INFO: Block design created successfully"

# =============================================================
# Generate HDL Wrapper
# =============================================================
make_wrapper -files [get_files ${bd_name}.bd] -top
set wrapper_file [file normalize ./$project_name/${project_name}.gen/sources_1/bd/$bd_name/hdl/${bd_name}_wrapper.v]
if {![file exists $wrapper_file]} {
    set wrapper_file [file normalize ./$project_name/${project_name}.srcs/sources_1/bd/$bd_name/hdl/${bd_name}_wrapper.v]
}
add_files -norecurse $wrapper_file
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1
puts "INFO: HDL wrapper created"

# =============================================================
# Run Synthesis
# =============================================================
puts "INFO: Starting synthesis..."
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed"
    exit 1
}
puts "INFO: Synthesis completed"

# =============================================================
# Run Implementation
# =============================================================
puts "INFO: Starting implementation..."
launch_runs impl_1 -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed"
    exit 1
}
puts "INFO: Implementation completed"

# =============================================================
# Generate Bitstream
# =============================================================
puts "INFO: Generating bitstream..."
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
puts "INFO: Bitstream generation completed"

# =============================================================
# Copy Output Files
# =============================================================
set bit_src [file normalize ./$project_name/${project_name}.runs/impl_1/${bd_name}_wrapper.bit]
set bit_dst [file normalize ./${project_name}.bit]
if {[file exists $bit_src]} {
    file copy -force $bit_src $bit_dst
    puts "INFO: Bitstream: $bit_dst"
} else {
    puts "ERROR: Bitstream not found at $bit_src"
    exit 1
}

set hwh_pattern [file normalize ./$project_name/${project_name}.gen/sources_1/bd/$bd_name/hw_handoff/${bd_name}.hwh]
set hwh_files [glob -nocomplain $hwh_pattern]
if {$hwh_files eq ""} {
    set hwh_pattern [file normalize ./$project_name/${project_name}.srcs/sources_1/bd/$bd_name/hw_handoff/${bd_name}.hwh]
    set hwh_files [glob -nocomplain $hwh_pattern]
}
if {$hwh_files ne ""} {
    file copy -force [lindex $hwh_files 0] ./${project_name}.hwh
    puts "INFO: Hardware handoff: ${project_name}.hwh"
} else {
    puts "WARNING: HWH file not found - PYNQ overlay loading may fail"
}

write_hw_platform -fixed -include_bit -force ./${project_name}.xsa
puts "INFO: XSA exported: ${project_name}.xsa"

puts "============================================================="
puts " BUILD COMPLETE"
puts " Bitstream: ${project_name}.bit"
puts " Hardware:  ${project_name}.hwh"
puts " XSA:       ${project_name}.xsa"
puts "============================================================="
