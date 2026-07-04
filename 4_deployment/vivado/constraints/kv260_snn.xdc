# =============================================================
# KV260 SNN Accelerator Constraints
# =============================================================
# FPGA Part: xck26-sfvc784-2LV-c
# Target Clock: 100MHz (10ns period)
# =============================================================

# Bitstream configuration
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.OVERTEMPSHUTDOWN ENABLE [current_design]

# Suppress DRC warnings for unconstrained ports (handled by PS)
set_property SEVERITY {Warning} [get_drc_checks NSTD-1]
set_property SEVERITY {Warning} [get_drc_checks UCIO-1]

# Clock constraint (should match PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ)
# 100MHz = 10ns period
# create_clock -period 10.000 -name pl_clk0 [get_pins zynq_ps/pl_clk0]
# Note: Clock is usually auto-constrained by PS preset, uncomment if needed
