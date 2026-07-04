# =============================================================
# SNN ResNet-18 Block Design for KRIA KV260
# =============================================================
# FPGA Part: xck26-sfvc784-2LV-c (Zynq UltraScale+ K26 SOM)
# Board Part: xilinx.com:kv260_som:part0:1.4
#
# Architecture:
#   PS GP0 -> SmartConnect -> HLS Control (s_axi_control + s_axi_control_r)
#   HLS m_axi_gmem0 -> SmartConnect -> PS HP0 (input + output)
#   HLS m_axi_gmem1 -> SmartConnect -> PS HP1 (weights)
#
# NO DMA needed - HLS reads/writes DDR directly via m_axi
# =============================================================

proc create_snn_bd {design_name} {
    create_bd_design $design_name

    # ================================================================
    # Create Zynq UltraScale+ MPSoC PS
    # ================================================================
    create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.4 zynq_ps

    set_property -dict [list \
        CONFIG.PSU__USE__M_AXI_GP0 {1} \
        CONFIG.PSU__USE__M_AXI_GP1 {0} \
        CONFIG.PSU__USE__M_AXI_GP2 {0} \
        CONFIG.PSU__USE__S_AXI_GP0 {0} \
        CONFIG.PSU__USE__S_AXI_GP1 {0} \
        CONFIG.PSU__USE__S_AXI_GP2 {1} \
        CONFIG.PSU__USE__S_AXI_GP3 {1} \
        CONFIG.PSU__USE__S_AXI_GP4 {0} \
        CONFIG.PSU__USE__S_AXI_GP5 {0} \
        CONFIG.PSU__USE__S_AXI_GP6 {0} \
        CONFIG.PSU__USE__IRQ0 {0} \
        CONFIG.PSU__FPGA_PL0_ENABLE {1} \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
        CONFIG.PSU__FPGA_PL1_ENABLE {0} \
    ] [get_bd_cells zynq_ps]

    # Connect PL clocks to AXI interface clocks
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins zynq_ps/maxihpm0_fpd_aclk]
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins zynq_ps/saxihp0_fpd_aclk]
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins zynq_ps/saxihp1_fpd_aclk]

    # ================================================================
    # Create Processor System Reset
    # ================================================================
    create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 ps_reset
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins ps_reset/slowest_sync_clk]
    connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins ps_reset/ext_reset_in]

    # ================================================================
    # SmartConnect GP0: PS Master -> HLS control AXI-Lite ports
    # NUM_MI depends on whether HLS generates 1 or 2 control slaves
    # Default: 2 (s_axi_control + s_axi_control_r)
    # If HLS generates only 1 slave, change NUM_MI to 1
    # ================================================================
    create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_gp0
    set_property -dict [list \
        CONFIG.NUM_SI {1} \
        CONFIG.NUM_MI {2} \
    ] [get_bd_cells axi_smc_gp0]

    connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD] \
        [get_bd_intf_pins axi_smc_gp0/S00_AXI]
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins axi_smc_gp0/aclk]
    connect_bd_net [get_bd_pins ps_reset/peripheral_aresetn] [get_bd_pins axi_smc_gp0/aresetn]

    # ================================================================
    # SmartConnect HP0: HLS gmem0 -> PS S_AXI_HP0_FPD
    # ================================================================
    create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_hp0
    set_property -dict [list \
        CONFIG.NUM_SI {1} \
        CONFIG.NUM_MI {1} \
    ] [get_bd_cells axi_smc_hp0]

    connect_bd_intf_net [get_bd_intf_pins axi_smc_hp0/M00_AXI] \
        [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins axi_smc_hp0/aclk]
    connect_bd_net [get_bd_pins ps_reset/peripheral_aresetn] [get_bd_pins axi_smc_hp0/aresetn]

    # ================================================================
    # SmartConnect HP1: HLS gmem1 -> PS S_AXI_HP1_FPD
    # ================================================================
    create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_hp1
    set_property -dict [list \
        CONFIG.NUM_SI {1} \
        CONFIG.NUM_MI {1} \
    ] [get_bd_cells axi_smc_hp1]

    connect_bd_intf_net [get_bd_intf_pins axi_smc_hp1/M00_AXI] \
        [get_bd_intf_pins zynq_ps/S_AXI_HP1_FPD]
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins axi_smc_hp1/aclk]
    connect_bd_net [get_bd_pins ps_reset/peripheral_aresetn] [get_bd_pins axi_smc_hp1/aresetn]

    # ================================================================
    # Create HLS SNN IP
    # ================================================================
    create_bd_cell -type ip -vlnv custom:hls:snn_top:1.0 snn_top_0

    # HLS clock and reset
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins snn_top_0/ap_clk]
    connect_bd_net [get_bd_pins ps_reset/peripheral_aresetn] [get_bd_pins snn_top_0/ap_rst_n]

    # ================================================================
    # Connect HLS AXI-Lite Control Ports -> SmartConnect GP0
    # ================================================================
    # M00 -> s_axi_control (ap_start, ap_done, ap_return)
    connect_bd_intf_net [get_bd_intf_pins axi_smc_gp0/M00_AXI] \
        [get_bd_intf_pins snn_top_0/s_axi_control]

    # M01 -> s_axi_control_r (input_r, weights, output_r addresses)
    # NOTE: If HLS generates only 1 control slave, comment out this line
    #       and change NUM_MI to 1 above
    connect_bd_intf_net [get_bd_intf_pins axi_smc_gp0/M01_AXI] \
        [get_bd_intf_pins snn_top_0/s_axi_control_r]

    # ================================================================
    # Connect HLS AXI Master Ports -> SmartConnect HP0/HP1 -> PS DDR
    # ================================================================
    # gmem0 (input + output) -> HP0
    connect_bd_intf_net [get_bd_intf_pins snn_top_0/m_axi_gmem0] \
        [get_bd_intf_pins axi_smc_hp0/S00_AXI]
    # gmem1 (weights) -> HP1
    connect_bd_intf_net [get_bd_intf_pins snn_top_0/m_axi_gmem1] \
        [get_bd_intf_pins axi_smc_hp1/S00_AXI]

    # ================================================================
    # Address Assignment
    # ================================================================
    assign_bd_address

    puts "INFO: Address map assigned."

    # ================================================================
    # Validate and Save
    # ================================================================
    regenerate_bd_layout
    validate_bd_design
    save_bd_design
}
