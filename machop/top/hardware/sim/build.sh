
#!/bin/bash
# This script is used to build the verilator simulation
verilator top/hardware/rtl/fc3_weight.sv top/hardware/rtl/top.sv top/hardware/rtl/fc1_weight.sv top/hardware/rtl/fc3_bias.sv top/hardware/rtl/fc2_weight.sv top/hardware/rtl/fc1_bias.sv top/hardware/rtl/fc2_bias.sv top/hardware/sim/verilog/fc3_weight.sv top/hardware/sim/verilog/top_tb.sv top/hardware/sim/verilog/top_data_out_fifo.sv top/hardware/sim/verilog/top.sv top/hardware/sim/verilog/top_data_in_fifo.sv top/hardware/sim/verilog/fc1_weight.sv top/hardware/sim/verilog/fc3_bias.sv top/hardware/sim/verilog/fc2_weight.sv top/hardware/sim/verilog/fc1_bias.sv top/hardware/sim/verilog/fc2_bias.sv top/hardware/sim/verilog/fifo_para.v 
