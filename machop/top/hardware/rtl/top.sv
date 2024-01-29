
// =====================================
//     Mase Hardware
//     Model: top
//     29/01/2024 11:26:47
// =====================================
`timescale 1ns/1ps
module top #(
    parameter fc1_DATA_IN_0_PRECISION_0 = 32,
    parameter fc1_DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter fc1_DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter fc1_DATA_IN_0_TENSOR_SIZE_DIM_1 = 784,
    parameter fc1_DATA_IN_0_PARALLELISM_DIM_1 = 784,
    parameter fc1_DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter fc1_DATA_IN_0_PARALLELISM_DIM_2 = 1,
    parameter fc1_WEIGHT_PRECISION_0 = 32,
    parameter fc1_WEIGHT_TENSOR_SIZE_DIM_0 = 784,
    parameter fc1_WEIGHT_PARALLELISM_DIM_0 = 784,
    parameter fc1_WEIGHT_TENSOR_SIZE_DIM_1 = 784,
    parameter fc1_WEIGHT_PARALLELISM_DIM_1 = 784,
    parameter fc1_WEIGHT_TENSOR_SIZE_DIM_2 = 1,
    parameter fc1_WEIGHT_PARALLELISM_DIM_2 = 1,
    parameter fc1_BIAS_PRECISION_0 = 32,
    parameter fc1_BIAS_TENSOR_SIZE_DIM_0 = 784,
    parameter fc1_BIAS_PARALLELISM_DIM_0 = 784,
    parameter fc1_BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter fc1_BIAS_PARALLELISM_DIM_1 = 1,
    parameter fc1_BIAS_TENSOR_SIZE_DIM_2 = 1,
    parameter fc1_BIAS_PARALLELISM_DIM_2 = 1,
    parameter fc1_DATA_OUT_0_PRECISION_0 = 32,
    parameter fc1_DATA_OUT_0_TENSOR_SIZE_0_DIM_0 = 1,
    parameter fc1_DATA_OUT_0_PARALLELISM_0_DIM_0 = 1,
    parameter fc1_DATA_OUT_0_TENSOR_SIZE_0_DIM_1 = 784,
    parameter fc1_DATA_OUT_0_PARALLELISM_0_DIM_1 = 784,
    parameter fc1_DATA_OUT_0_TENSOR_SIZE_0_DIM_2 = 1,
    parameter fc1_DATA_OUT_0_PARALLELISM_0_DIM_2 = 1,
    parameter relu_DATA_IN_0_PRECISION_0 = 32,
    parameter relu_DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter relu_DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter relu_DATA_IN_0_TENSOR_SIZE_DIM_1 = 784,
    parameter relu_DATA_IN_0_PARALLELISM_DIM_1 = 784,
    parameter relu_DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter relu_DATA_IN_0_PARALLELISM_DIM_2 = 1,
    parameter relu_INPLACE = False,
    parameter relu_DATA_OUT_0_PRECISION_0 = 32,
    parameter relu_DATA_OUT_0_TENSOR_SIZE_0_DIM_0 = 1,
    parameter relu_DATA_OUT_0_PARALLELISM_0_DIM_0 = 1,
    parameter relu_DATA_OUT_0_TENSOR_SIZE_0_DIM_1 = 784,
    parameter relu_DATA_OUT_0_PARALLELISM_0_DIM_1 = 784,
    parameter relu_DATA_OUT_0_TENSOR_SIZE_0_DIM_2 = 1,
    parameter relu_DATA_OUT_0_PARALLELISM_0_DIM_2 = 1,
    parameter fc2_DATA_IN_0_PRECISION_0 = 32,
    parameter fc2_DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter fc2_DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter fc2_DATA_IN_0_TENSOR_SIZE_DIM_1 = 784,
    parameter fc2_DATA_IN_0_PARALLELISM_DIM_1 = 784,
    parameter fc2_DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter fc2_DATA_IN_0_PARALLELISM_DIM_2 = 1,
    parameter fc2_WEIGHT_PRECISION_0 = 32,
    parameter fc2_WEIGHT_TENSOR_SIZE_DIM_0 = 3136,
    parameter fc2_WEIGHT_PARALLELISM_DIM_0 = 3136,
    parameter fc2_WEIGHT_TENSOR_SIZE_DIM_1 = 784,
    parameter fc2_WEIGHT_PARALLELISM_DIM_1 = 784,
    parameter fc2_WEIGHT_TENSOR_SIZE_DIM_2 = 1,
    parameter fc2_WEIGHT_PARALLELISM_DIM_2 = 1,
    parameter fc2_BIAS_PRECISION_0 = 32,
    parameter fc2_BIAS_TENSOR_SIZE_DIM_0 = 3136,
    parameter fc2_BIAS_PARALLELISM_DIM_0 = 3136,
    parameter fc2_BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter fc2_BIAS_PARALLELISM_DIM_1 = 1,
    parameter fc2_BIAS_TENSOR_SIZE_DIM_2 = 1,
    parameter fc2_BIAS_PARALLELISM_DIM_2 = 1,
    parameter fc2_DATA_OUT_0_PRECISION_0 = 32,
    parameter fc2_DATA_OUT_0_TENSOR_SIZE_0_DIM_0 = 1,
    parameter fc2_DATA_OUT_0_PARALLELISM_0_DIM_0 = 1,
    parameter fc2_DATA_OUT_0_TENSOR_SIZE_0_DIM_1 = 3136,
    parameter fc2_DATA_OUT_0_PARALLELISM_0_DIM_1 = 3136,
    parameter fc2_DATA_OUT_0_TENSOR_SIZE_0_DIM_2 = 1,
    parameter fc2_DATA_OUT_0_PARALLELISM_0_DIM_2 = 1,
    parameter relu_1_DATA_IN_0_PRECISION_0 = 32,
    parameter relu_1_DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter relu_1_DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter relu_1_DATA_IN_0_TENSOR_SIZE_DIM_1 = 3136,
    parameter relu_1_DATA_IN_0_PARALLELISM_DIM_1 = 3136,
    parameter relu_1_DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter relu_1_DATA_IN_0_PARALLELISM_DIM_2 = 1,
    parameter relu_1_INPLACE = False,
    parameter relu_1_DATA_OUT_0_PRECISION_0 = 32,
    parameter relu_1_DATA_OUT_0_TENSOR_SIZE_0_DIM_0 = 1,
    parameter relu_1_DATA_OUT_0_PARALLELISM_0_DIM_0 = 1,
    parameter relu_1_DATA_OUT_0_TENSOR_SIZE_0_DIM_1 = 3136,
    parameter relu_1_DATA_OUT_0_PARALLELISM_0_DIM_1 = 3136,
    parameter relu_1_DATA_OUT_0_TENSOR_SIZE_0_DIM_2 = 1,
    parameter relu_1_DATA_OUT_0_PARALLELISM_0_DIM_2 = 1,
    parameter fc3_DATA_IN_0_PRECISION_0 = 32,
    parameter fc3_DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter fc3_DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter fc3_DATA_IN_0_TENSOR_SIZE_DIM_1 = 3136,
    parameter fc3_DATA_IN_0_PARALLELISM_DIM_1 = 3136,
    parameter fc3_DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter fc3_DATA_IN_0_PARALLELISM_DIM_2 = 1,
    parameter fc3_WEIGHT_PRECISION_0 = 32,
    parameter fc3_WEIGHT_TENSOR_SIZE_DIM_0 = 10,
    parameter fc3_WEIGHT_PARALLELISM_DIM_0 = 10,
    parameter fc3_WEIGHT_TENSOR_SIZE_DIM_1 = 3136,
    parameter fc3_WEIGHT_PARALLELISM_DIM_1 = 3136,
    parameter fc3_WEIGHT_TENSOR_SIZE_DIM_2 = 1,
    parameter fc3_WEIGHT_PARALLELISM_DIM_2 = 1,
    parameter fc3_BIAS_PRECISION_0 = 32,
    parameter fc3_BIAS_TENSOR_SIZE_DIM_0 = 10,
    parameter fc3_BIAS_PARALLELISM_DIM_0 = 10,
    parameter fc3_BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter fc3_BIAS_PARALLELISM_DIM_1 = 1,
    parameter fc3_BIAS_TENSOR_SIZE_DIM_2 = 1,
    parameter fc3_BIAS_PARALLELISM_DIM_2 = 1,
    parameter fc3_DATA_OUT_0_PRECISION_0 = 32,
    parameter fc3_DATA_OUT_0_TENSOR_SIZE_0_DIM_0 = 1,
    parameter fc3_DATA_OUT_0_PARALLELISM_0_DIM_0 = 1,
    parameter fc3_DATA_OUT_0_TENSOR_SIZE_0_DIM_1 = 10,
    parameter fc3_DATA_OUT_0_PARALLELISM_0_DIM_1 = 10,
    parameter fc3_DATA_OUT_0_TENSOR_SIZE_0_DIM_2 = 1,
    parameter fc3_DATA_OUT_0_PARALLELISM_0_DIM_2 = 1
) (
    input clk,
    input rst,

    input  [fc1_DATA_IN_0_PRECISION_0-1:0] data_in_0 [fc1_DATA_IN_0_PARALLELISM_DIM_2*fc1_DATA_IN_0_PARALLELISM_DIM_1*fc1_DATA_IN_0_PARALLELISM_DIM_0-1:0],
    input  data_in_0_valid,
    output data_in_0_ready,
    output  [fc3_DATA_OUT_0_PRECISION_0-1:0] data_out_0 [fc3_DATA_OUT_0_PARALLELISM_DIM_2*fc3_DATA_OUT_0_PARALLELISM_DIM_1*fc3_DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    output  data_out_0_valid,
    input data_out_0_ready
);

// --------------------------
//   fc1 signals
// --------------------------
logic [fc1_IN_0_PRECISION_0-1:0]  fc1_data_in_0        [fc1_IN_0_TENSOR_SIZE_DIM_0-1:0];
logic                             fc1_data_in_0_valid;
logic                             fc1_data_in_0_ready;
logic [fc1_WEIGHT_PRECISION_0-1:0]  fc1_weight        [fc1_WEIGHT_TENSOR_SIZE_DIM_0-1:0];
logic                             fc1_weight_valid;
logic                             fc1_weight_ready;
logic [fc1_BIAS_PRECISION_0-1:0]  fc1_bias        [fc1_BIAS_TENSOR_SIZE_DIM_0-1:0];
logic                             fc1_bias_valid;
logic                             fc1_bias_ready;
logic [fc1_OUT_0_PRECISION_0-1:0]  fc1_data_out_0        [fc1_OUT_0_TENSOR_SIZE_DIM_0-1:0];
logic                             fc1_data_out_0_valid;
logic                             fc1_data_out_0_ready;
// --------------------------
//   relu signals
// --------------------------
logic [relu_IN_0_PRECISION_0-1:0]  relu_data_in_0        [relu_IN_0_TENSOR_SIZE_DIM_0-1:0];
logic                             relu_data_in_0_valid;
logic                             relu_data_in_0_ready;
logic [relu_OUT_0_PRECISION_0-1:0]  relu_data_out_0        [relu_OUT_0_TENSOR_SIZE_DIM_0-1:0];
logic                             relu_data_out_0_valid;
logic                             relu_data_out_0_ready;
// --------------------------
//   fc2 signals
// --------------------------
logic [fc2_IN_0_PRECISION_0-1:0]  fc2_data_in_0        [fc2_IN_0_TENSOR_SIZE_DIM_0-1:0];
logic                             fc2_data_in_0_valid;
logic                             fc2_data_in_0_ready;
logic [fc2_WEIGHT_PRECISION_0-1:0]  fc2_weight        [fc2_WEIGHT_TENSOR_SIZE_DIM_0-1:0];
logic                             fc2_weight_valid;
logic                             fc2_weight_ready;
logic [fc2_BIAS_PRECISION_0-1:0]  fc2_bias        [fc2_BIAS_TENSOR_SIZE_DIM_0-1:0];
logic                             fc2_bias_valid;
logic                             fc2_bias_ready;
logic [fc2_OUT_0_PRECISION_0-1:0]  fc2_data_out_0        [fc2_OUT_0_TENSOR_SIZE_DIM_0-1:0];
logic                             fc2_data_out_0_valid;
logic                             fc2_data_out_0_ready;
// --------------------------
//   relu_1 signals
// --------------------------
logic [relu_1_IN_0_PRECISION_0-1:0]  relu_1_data_in_0        [relu_1_IN_0_TENSOR_SIZE_DIM_0-1:0];
logic                             relu_1_data_in_0_valid;
logic                             relu_1_data_in_0_ready;
logic [relu_1_OUT_0_PRECISION_0-1:0]  relu_1_data_out_0        [relu_1_OUT_0_TENSOR_SIZE_DIM_0-1:0];
logic                             relu_1_data_out_0_valid;
logic                             relu_1_data_out_0_ready;
// --------------------------
//   fc3 signals
// --------------------------
logic [fc3_IN_0_PRECISION_0-1:0]  fc3_data_in_0        [fc3_IN_0_TENSOR_SIZE_DIM_0-1:0];
logic                             fc3_data_in_0_valid;
logic                             fc3_data_in_0_ready;
logic [fc3_WEIGHT_PRECISION_0-1:0]  fc3_weight        [fc3_WEIGHT_TENSOR_SIZE_DIM_0-1:0];
logic                             fc3_weight_valid;
logic                             fc3_weight_ready;
logic [fc3_BIAS_PRECISION_0-1:0]  fc3_bias        [fc3_BIAS_TENSOR_SIZE_DIM_0-1:0];
logic                             fc3_bias_valid;
logic                             fc3_bias_ready;
logic [fc3_OUT_0_PRECISION_0-1:0]  fc3_data_out_0        [fc3_OUT_0_TENSOR_SIZE_DIM_0-1:0];
logic                             fc3_data_out_0_valid;
logic                             fc3_data_out_0_ready;

// --------------------------
//   Component instantiation
// --------------------------

// fc1
fixed_linear #(
    .DATA_IN_0_PRECISION_0(fc1_DATA_IN_0_PRECISION_0), // = 32
    .DATA_IN_0_TENSOR_SIZE_DIM_0(fc1_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 1
    .DATA_IN_0_PARALLELISM_DIM_0(fc1_DATA_IN_0_PARALLELISM_DIM_0), // = 1
    .DATA_IN_0_TENSOR_SIZE_DIM_1(fc1_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 784
    .DATA_IN_0_PARALLELISM_DIM_1(fc1_DATA_IN_0_PARALLELISM_DIM_1), // = 784
    .DATA_IN_0_TENSOR_SIZE_DIM_2(fc1_DATA_IN_0_TENSOR_SIZE_DIM_2), // = 1
    .DATA_IN_0_PARALLELISM_DIM_2(fc1_DATA_IN_0_PARALLELISM_DIM_2), // = 1
    .WEIGHT_PRECISION_0(fc1_WEIGHT_PRECISION_0), // = 32
    .WEIGHT_TENSOR_SIZE_DIM_0(fc1_WEIGHT_TENSOR_SIZE_DIM_0), // = 784
    .WEIGHT_PARALLELISM_DIM_0(fc1_WEIGHT_PARALLELISM_DIM_0), // = 784
    .WEIGHT_TENSOR_SIZE_DIM_1(fc1_WEIGHT_TENSOR_SIZE_DIM_1), // = 784
    .WEIGHT_PARALLELISM_DIM_1(fc1_WEIGHT_PARALLELISM_DIM_1), // = 784
    .WEIGHT_TENSOR_SIZE_DIM_2(fc1_WEIGHT_TENSOR_SIZE_DIM_2), // = 1
    .WEIGHT_PARALLELISM_DIM_2(fc1_WEIGHT_PARALLELISM_DIM_2), // = 1
    .BIAS_PRECISION_0(fc1_BIAS_PRECISION_0), // = 32
    .BIAS_TENSOR_SIZE_DIM_0(fc1_BIAS_TENSOR_SIZE_DIM_0), // = 784
    .BIAS_PARALLELISM_DIM_0(fc1_BIAS_PARALLELISM_DIM_0), // = 784
    .BIAS_TENSOR_SIZE_DIM_1(fc1_BIAS_TENSOR_SIZE_DIM_1), // = 1
    .BIAS_PARALLELISM_DIM_1(fc1_BIAS_PARALLELISM_DIM_1), // = 1
    .BIAS_TENSOR_SIZE_DIM_2(fc1_BIAS_TENSOR_SIZE_DIM_2), // = 1
    .BIAS_PARALLELISM_DIM_2(fc1_BIAS_PARALLELISM_DIM_2), // = 1
    .DATA_OUT_0_PRECISION_0(fc1_DATA_OUT_0_PRECISION_0), // = 32
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_0(fc1_DATA_OUT_0_TENSOR_SIZE_0_DIM_0), // = 1
    .DATA_OUT_0_PARALLELISM_0_DIM_0(fc1_DATA_OUT_0_PARALLELISM_0_DIM_0), // = 1
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_1(fc1_DATA_OUT_0_TENSOR_SIZE_0_DIM_1), // = 784
    .DATA_OUT_0_PARALLELISM_0_DIM_1(fc1_DATA_OUT_0_PARALLELISM_0_DIM_1), // = 784
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_2(fc1_DATA_OUT_0_TENSOR_SIZE_0_DIM_2), // = 1
    .DATA_OUT_0_PARALLELISM_0_DIM_2(fc1_DATA_OUT_0_PARALLELISM_0_DIM_2)
) fc1_inst (
    .clk(clk),
    .rst(rst),

    .data_in_0(fc1_data_in_0),
    .data_in_0_valid(fc1_data_in_0_valid),
    .data_in_0_ready(fc1_data_in_0_ready),
    
    .weight(fc1_weight),
    .weight_valid(fc1_weight_valid),
    .weight_ready(fc1_weight_ready),
    
    .bias(fc1_bias),
    .bias_valid(fc1_bias_valid),
    .bias_ready(fc1_bias_ready),
    
    .data_out_0(fc1_data_out_0),
    .data_out_0_valid(fc1_data_out_0_valid),
    .data_out_0_ready(fc1_data_out_0_ready)
);

fc1_weight_source #(
.WEIGHT_PRECISION_0(fc1_WEIGHT_PRECISION_0),
.WEIGHT_TENSOR_SIZE_DIM_0(fc1_WEIGHT_TENSOR_SIZE_DIM_0),
.WEIGHT_PARALLELISM_DIM_0(fc1_WEIGHT_PARALLELISM_DIM_0),
.WEIGHT_TENSOR_SIZE_DIM_1(fc1_WEIGHT_TENSOR_SIZE_DIM_1),
.WEIGHT_PARALLELISM_DIM_1(fc1_WEIGHT_PARALLELISM_DIM_1),
.WEIGHT_TENSOR_SIZE_DIM_2(fc1_WEIGHT_TENSOR_SIZE_DIM_2),
.WEIGHT_PARALLELISM_DIM_2(fc1_WEIGHT_PARALLELISM_DIM_2)
) fc1_weight_source_0 (
    .clk(clk),
    .rst(rst),
    .data_out(fc1_weight),
    .data_out_ready(fc1_weight_ready),
    .data_out_valid(fc1_weight_valid)
);

fc1_bias_source #(
.BIAS_PRECISION_0(fc1_BIAS_PRECISION_0),
.BIAS_TENSOR_SIZE_DIM_0(fc1_BIAS_TENSOR_SIZE_DIM_0),
.BIAS_PARALLELISM_DIM_0(fc1_BIAS_PARALLELISM_DIM_0),
.BIAS_TENSOR_SIZE_DIM_1(fc1_BIAS_TENSOR_SIZE_DIM_1),
.BIAS_PARALLELISM_DIM_1(fc1_BIAS_PARALLELISM_DIM_1),
.BIAS_TENSOR_SIZE_DIM_2(fc1_BIAS_TENSOR_SIZE_DIM_2),
.BIAS_PARALLELISM_DIM_2(fc1_BIAS_PARALLELISM_DIM_2)
) fc1_bias_source_0 (
    .clk(clk),
    .rst(rst),
    .data_out(fc1_bias),
    .data_out_ready(fc1_bias_ready),
    .data_out_valid(fc1_bias_valid)
);

// relu
fixed_relu #(
    .DATA_IN_0_PRECISION_0(relu_DATA_IN_0_PRECISION_0), // = 32
    .DATA_IN_0_TENSOR_SIZE_DIM_0(relu_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 1
    .DATA_IN_0_PARALLELISM_DIM_0(relu_DATA_IN_0_PARALLELISM_DIM_0), // = 1
    .DATA_IN_0_TENSOR_SIZE_DIM_1(relu_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 784
    .DATA_IN_0_PARALLELISM_DIM_1(relu_DATA_IN_0_PARALLELISM_DIM_1), // = 784
    .DATA_IN_0_TENSOR_SIZE_DIM_2(relu_DATA_IN_0_TENSOR_SIZE_DIM_2), // = 1
    .DATA_IN_0_PARALLELISM_DIM_2(relu_DATA_IN_0_PARALLELISM_DIM_2), // = 1
    .INPLACE(relu_INPLACE), // = False
    .DATA_OUT_0_PRECISION_0(relu_DATA_OUT_0_PRECISION_0), // = 32
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_0(relu_DATA_OUT_0_TENSOR_SIZE_0_DIM_0), // = 1
    .DATA_OUT_0_PARALLELISM_0_DIM_0(relu_DATA_OUT_0_PARALLELISM_0_DIM_0), // = 1
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_1(relu_DATA_OUT_0_TENSOR_SIZE_0_DIM_1), // = 784
    .DATA_OUT_0_PARALLELISM_0_DIM_1(relu_DATA_OUT_0_PARALLELISM_0_DIM_1), // = 784
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_2(relu_DATA_OUT_0_TENSOR_SIZE_0_DIM_2), // = 1
    .DATA_OUT_0_PARALLELISM_0_DIM_2(relu_DATA_OUT_0_PARALLELISM_0_DIM_2)
) relu_inst (
    .clk(clk),
    .rst(rst),

    .data_in_0(relu_data_in_0),
    .data_in_0_valid(relu_data_in_0_valid),
    .data_in_0_ready(relu_data_in_0_ready),
    
    .inplace(relu_inplace),
    .inplace_valid(relu_inplace_valid),
    .inplace_ready(relu_inplace_ready),
    
    .data_out_0(relu_data_out_0),
    .data_out_0_valid(relu_data_out_0_valid),
    .data_out_0_ready(relu_data_out_0_ready)
);

// fc2
fixed_linear #(
    .DATA_IN_0_PRECISION_0(fc2_DATA_IN_0_PRECISION_0), // = 32
    .DATA_IN_0_TENSOR_SIZE_DIM_0(fc2_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 1
    .DATA_IN_0_PARALLELISM_DIM_0(fc2_DATA_IN_0_PARALLELISM_DIM_0), // = 1
    .DATA_IN_0_TENSOR_SIZE_DIM_1(fc2_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 784
    .DATA_IN_0_PARALLELISM_DIM_1(fc2_DATA_IN_0_PARALLELISM_DIM_1), // = 784
    .DATA_IN_0_TENSOR_SIZE_DIM_2(fc2_DATA_IN_0_TENSOR_SIZE_DIM_2), // = 1
    .DATA_IN_0_PARALLELISM_DIM_2(fc2_DATA_IN_0_PARALLELISM_DIM_2), // = 1
    .WEIGHT_PRECISION_0(fc2_WEIGHT_PRECISION_0), // = 32
    .WEIGHT_TENSOR_SIZE_DIM_0(fc2_WEIGHT_TENSOR_SIZE_DIM_0), // = 3136
    .WEIGHT_PARALLELISM_DIM_0(fc2_WEIGHT_PARALLELISM_DIM_0), // = 3136
    .WEIGHT_TENSOR_SIZE_DIM_1(fc2_WEIGHT_TENSOR_SIZE_DIM_1), // = 784
    .WEIGHT_PARALLELISM_DIM_1(fc2_WEIGHT_PARALLELISM_DIM_1), // = 784
    .WEIGHT_TENSOR_SIZE_DIM_2(fc2_WEIGHT_TENSOR_SIZE_DIM_2), // = 1
    .WEIGHT_PARALLELISM_DIM_2(fc2_WEIGHT_PARALLELISM_DIM_2), // = 1
    .BIAS_PRECISION_0(fc2_BIAS_PRECISION_0), // = 32
    .BIAS_TENSOR_SIZE_DIM_0(fc2_BIAS_TENSOR_SIZE_DIM_0), // = 3136
    .BIAS_PARALLELISM_DIM_0(fc2_BIAS_PARALLELISM_DIM_0), // = 3136
    .BIAS_TENSOR_SIZE_DIM_1(fc2_BIAS_TENSOR_SIZE_DIM_1), // = 1
    .BIAS_PARALLELISM_DIM_1(fc2_BIAS_PARALLELISM_DIM_1), // = 1
    .BIAS_TENSOR_SIZE_DIM_2(fc2_BIAS_TENSOR_SIZE_DIM_2), // = 1
    .BIAS_PARALLELISM_DIM_2(fc2_BIAS_PARALLELISM_DIM_2), // = 1
    .DATA_OUT_0_PRECISION_0(fc2_DATA_OUT_0_PRECISION_0), // = 32
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_0(fc2_DATA_OUT_0_TENSOR_SIZE_0_DIM_0), // = 1
    .DATA_OUT_0_PARALLELISM_0_DIM_0(fc2_DATA_OUT_0_PARALLELISM_0_DIM_0), // = 1
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_1(fc2_DATA_OUT_0_TENSOR_SIZE_0_DIM_1), // = 3136
    .DATA_OUT_0_PARALLELISM_0_DIM_1(fc2_DATA_OUT_0_PARALLELISM_0_DIM_1), // = 3136
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_2(fc2_DATA_OUT_0_TENSOR_SIZE_0_DIM_2), // = 1
    .DATA_OUT_0_PARALLELISM_0_DIM_2(fc2_DATA_OUT_0_PARALLELISM_0_DIM_2)
) fc2_inst (
    .clk(clk),
    .rst(rst),

    .data_in_0(fc2_data_in_0),
    .data_in_0_valid(fc2_data_in_0_valid),
    .data_in_0_ready(fc2_data_in_0_ready),
    
    .weight(fc2_weight),
    .weight_valid(fc2_weight_valid),
    .weight_ready(fc2_weight_ready),
    
    .bias(fc2_bias),
    .bias_valid(fc2_bias_valid),
    .bias_ready(fc2_bias_ready),
    
    .data_out_0(fc2_data_out_0),
    .data_out_0_valid(fc2_data_out_0_valid),
    .data_out_0_ready(fc2_data_out_0_ready)
);

fc2_weight_source #(
.WEIGHT_PRECISION_0(fc2_WEIGHT_PRECISION_0),
.WEIGHT_TENSOR_SIZE_DIM_0(fc2_WEIGHT_TENSOR_SIZE_DIM_0),
.WEIGHT_PARALLELISM_DIM_0(fc2_WEIGHT_PARALLELISM_DIM_0),
.WEIGHT_TENSOR_SIZE_DIM_1(fc2_WEIGHT_TENSOR_SIZE_DIM_1),
.WEIGHT_PARALLELISM_DIM_1(fc2_WEIGHT_PARALLELISM_DIM_1),
.WEIGHT_TENSOR_SIZE_DIM_2(fc2_WEIGHT_TENSOR_SIZE_DIM_2),
.WEIGHT_PARALLELISM_DIM_2(fc2_WEIGHT_PARALLELISM_DIM_2)
) fc2_weight_source_0 (
    .clk(clk),
    .rst(rst),
    .data_out(fc2_weight),
    .data_out_ready(fc2_weight_ready),
    .data_out_valid(fc2_weight_valid)
);

fc2_bias_source #(
.BIAS_PRECISION_0(fc2_BIAS_PRECISION_0),
.BIAS_TENSOR_SIZE_DIM_0(fc2_BIAS_TENSOR_SIZE_DIM_0),
.BIAS_PARALLELISM_DIM_0(fc2_BIAS_PARALLELISM_DIM_0),
.BIAS_TENSOR_SIZE_DIM_1(fc2_BIAS_TENSOR_SIZE_DIM_1),
.BIAS_PARALLELISM_DIM_1(fc2_BIAS_PARALLELISM_DIM_1),
.BIAS_TENSOR_SIZE_DIM_2(fc2_BIAS_TENSOR_SIZE_DIM_2),
.BIAS_PARALLELISM_DIM_2(fc2_BIAS_PARALLELISM_DIM_2)
) fc2_bias_source_0 (
    .clk(clk),
    .rst(rst),
    .data_out(fc2_bias),
    .data_out_ready(fc2_bias_ready),
    .data_out_valid(fc2_bias_valid)
);

// relu_1
fixed_relu #(
    .DATA_IN_0_PRECISION_0(relu_1_DATA_IN_0_PRECISION_0), // = 32
    .DATA_IN_0_TENSOR_SIZE_DIM_0(relu_1_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 1
    .DATA_IN_0_PARALLELISM_DIM_0(relu_1_DATA_IN_0_PARALLELISM_DIM_0), // = 1
    .DATA_IN_0_TENSOR_SIZE_DIM_1(relu_1_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 3136
    .DATA_IN_0_PARALLELISM_DIM_1(relu_1_DATA_IN_0_PARALLELISM_DIM_1), // = 3136
    .DATA_IN_0_TENSOR_SIZE_DIM_2(relu_1_DATA_IN_0_TENSOR_SIZE_DIM_2), // = 1
    .DATA_IN_0_PARALLELISM_DIM_2(relu_1_DATA_IN_0_PARALLELISM_DIM_2), // = 1
    .INPLACE(relu_1_INPLACE), // = False
    .DATA_OUT_0_PRECISION_0(relu_1_DATA_OUT_0_PRECISION_0), // = 32
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_0(relu_1_DATA_OUT_0_TENSOR_SIZE_0_DIM_0), // = 1
    .DATA_OUT_0_PARALLELISM_0_DIM_0(relu_1_DATA_OUT_0_PARALLELISM_0_DIM_0), // = 1
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_1(relu_1_DATA_OUT_0_TENSOR_SIZE_0_DIM_1), // = 3136
    .DATA_OUT_0_PARALLELISM_0_DIM_1(relu_1_DATA_OUT_0_PARALLELISM_0_DIM_1), // = 3136
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_2(relu_1_DATA_OUT_0_TENSOR_SIZE_0_DIM_2), // = 1
    .DATA_OUT_0_PARALLELISM_0_DIM_2(relu_1_DATA_OUT_0_PARALLELISM_0_DIM_2)
) relu_1_inst (
    .clk(clk),
    .rst(rst),

    .data_in_0(relu_1_data_in_0),
    .data_in_0_valid(relu_1_data_in_0_valid),
    .data_in_0_ready(relu_1_data_in_0_ready),
    
    .inplace(relu_1_inplace),
    .inplace_valid(relu_1_inplace_valid),
    .inplace_ready(relu_1_inplace_ready),
    
    .data_out_0(relu_1_data_out_0),
    .data_out_0_valid(relu_1_data_out_0_valid),
    .data_out_0_ready(relu_1_data_out_0_ready)
);

// fc3
fixed_linear #(
    .DATA_IN_0_PRECISION_0(fc3_DATA_IN_0_PRECISION_0), // = 32
    .DATA_IN_0_TENSOR_SIZE_DIM_0(fc3_DATA_IN_0_TENSOR_SIZE_DIM_0), // = 1
    .DATA_IN_0_PARALLELISM_DIM_0(fc3_DATA_IN_0_PARALLELISM_DIM_0), // = 1
    .DATA_IN_0_TENSOR_SIZE_DIM_1(fc3_DATA_IN_0_TENSOR_SIZE_DIM_1), // = 3136
    .DATA_IN_0_PARALLELISM_DIM_1(fc3_DATA_IN_0_PARALLELISM_DIM_1), // = 3136
    .DATA_IN_0_TENSOR_SIZE_DIM_2(fc3_DATA_IN_0_TENSOR_SIZE_DIM_2), // = 1
    .DATA_IN_0_PARALLELISM_DIM_2(fc3_DATA_IN_0_PARALLELISM_DIM_2), // = 1
    .WEIGHT_PRECISION_0(fc3_WEIGHT_PRECISION_0), // = 32
    .WEIGHT_TENSOR_SIZE_DIM_0(fc3_WEIGHT_TENSOR_SIZE_DIM_0), // = 10
    .WEIGHT_PARALLELISM_DIM_0(fc3_WEIGHT_PARALLELISM_DIM_0), // = 10
    .WEIGHT_TENSOR_SIZE_DIM_1(fc3_WEIGHT_TENSOR_SIZE_DIM_1), // = 3136
    .WEIGHT_PARALLELISM_DIM_1(fc3_WEIGHT_PARALLELISM_DIM_1), // = 3136
    .WEIGHT_TENSOR_SIZE_DIM_2(fc3_WEIGHT_TENSOR_SIZE_DIM_2), // = 1
    .WEIGHT_PARALLELISM_DIM_2(fc3_WEIGHT_PARALLELISM_DIM_2), // = 1
    .BIAS_PRECISION_0(fc3_BIAS_PRECISION_0), // = 32
    .BIAS_TENSOR_SIZE_DIM_0(fc3_BIAS_TENSOR_SIZE_DIM_0), // = 10
    .BIAS_PARALLELISM_DIM_0(fc3_BIAS_PARALLELISM_DIM_0), // = 10
    .BIAS_TENSOR_SIZE_DIM_1(fc3_BIAS_TENSOR_SIZE_DIM_1), // = 1
    .BIAS_PARALLELISM_DIM_1(fc3_BIAS_PARALLELISM_DIM_1), // = 1
    .BIAS_TENSOR_SIZE_DIM_2(fc3_BIAS_TENSOR_SIZE_DIM_2), // = 1
    .BIAS_PARALLELISM_DIM_2(fc3_BIAS_PARALLELISM_DIM_2), // = 1
    .DATA_OUT_0_PRECISION_0(fc3_DATA_OUT_0_PRECISION_0), // = 32
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_0(fc3_DATA_OUT_0_TENSOR_SIZE_0_DIM_0), // = 1
    .DATA_OUT_0_PARALLELISM_0_DIM_0(fc3_DATA_OUT_0_PARALLELISM_0_DIM_0), // = 1
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_1(fc3_DATA_OUT_0_TENSOR_SIZE_0_DIM_1), // = 10
    .DATA_OUT_0_PARALLELISM_0_DIM_1(fc3_DATA_OUT_0_PARALLELISM_0_DIM_1), // = 10
    .DATA_OUT_0_TENSOR_SIZE_0_DIM_2(fc3_DATA_OUT_0_TENSOR_SIZE_0_DIM_2), // = 1
    .DATA_OUT_0_PARALLELISM_0_DIM_2(fc3_DATA_OUT_0_PARALLELISM_0_DIM_2)
) fc3_inst (
    .clk(clk),
    .rst(rst),

    .data_in_0(fc3_data_in_0),
    .data_in_0_valid(fc3_data_in_0_valid),
    .data_in_0_ready(fc3_data_in_0_ready),
    
    .weight(fc3_weight),
    .weight_valid(fc3_weight_valid),
    .weight_ready(fc3_weight_ready),
    
    .bias(fc3_bias),
    .bias_valid(fc3_bias_valid),
    .bias_ready(fc3_bias_ready),
    
    .data_out_0(fc3_data_out_0),
    .data_out_0_valid(fc3_data_out_0_valid),
    .data_out_0_ready(fc3_data_out_0_ready)
);

fc3_weight_source #(
.WEIGHT_PRECISION_0(fc3_WEIGHT_PRECISION_0),
.WEIGHT_TENSOR_SIZE_DIM_0(fc3_WEIGHT_TENSOR_SIZE_DIM_0),
.WEIGHT_PARALLELISM_DIM_0(fc3_WEIGHT_PARALLELISM_DIM_0),
.WEIGHT_TENSOR_SIZE_DIM_1(fc3_WEIGHT_TENSOR_SIZE_DIM_1),
.WEIGHT_PARALLELISM_DIM_1(fc3_WEIGHT_PARALLELISM_DIM_1),
.WEIGHT_TENSOR_SIZE_DIM_2(fc3_WEIGHT_TENSOR_SIZE_DIM_2),
.WEIGHT_PARALLELISM_DIM_2(fc3_WEIGHT_PARALLELISM_DIM_2)
) fc3_weight_source_0 (
    .clk(clk),
    .rst(rst),
    .data_out(fc3_weight),
    .data_out_ready(fc3_weight_ready),
    .data_out_valid(fc3_weight_valid)
);

fc3_bias_source #(
.BIAS_PRECISION_0(fc3_BIAS_PRECISION_0),
.BIAS_TENSOR_SIZE_DIM_0(fc3_BIAS_TENSOR_SIZE_DIM_0),
.BIAS_PARALLELISM_DIM_0(fc3_BIAS_PARALLELISM_DIM_0),
.BIAS_TENSOR_SIZE_DIM_1(fc3_BIAS_TENSOR_SIZE_DIM_1),
.BIAS_PARALLELISM_DIM_1(fc3_BIAS_PARALLELISM_DIM_1),
.BIAS_TENSOR_SIZE_DIM_2(fc3_BIAS_TENSOR_SIZE_DIM_2),
.BIAS_PARALLELISM_DIM_2(fc3_BIAS_PARALLELISM_DIM_2)
) fc3_bias_source_0 (
    .clk(clk),
    .rst(rst),
    .data_out(fc3_bias),
    .data_out_ready(fc3_bias_ready),
    .data_out_valid(fc3_bias_valid)
);


// --------------------------
//   Interconnections
// --------------------------
    
    assign data_in_0_ready = fc1_data_in_0_ready;
    assign fc1_data_in_0_valid    = data_in_0_valid;
    assign fc1_data_in_0    = data_in_0;

    assign data_out_0_valid = fc3_data_out_0_valid;
    assign fc3_data_out_0_ready    = data_out_0_ready;
    assign data_out_0 = fc3_data_out_0;

    assign fc1_data_out_0_ready  = relu_data_in_0_ready;
    assign relu_data_in_0_valid    = fc1_data_out_0_valid;
    assign relu_data_in_0 = fc1_data_out_0;

    assign relu_data_out_0_ready  = fc2_data_in_0_ready;
    assign fc2_data_in_0_valid    = relu_data_out_0_valid;
    assign fc2_data_in_0 = relu_data_out_0;

    assign fc2_data_out_0_ready  = relu_1_data_in_0_ready;
    assign relu_1_data_in_0_valid    = fc2_data_out_0_valid;
    assign relu_1_data_in_0 = fc2_data_out_0;

    assign relu_1_data_out_0_ready  = fc3_data_in_0_ready;
    assign fc3_data_in_0_valid    = relu_1_data_out_0_valid;
    assign fc3_data_in_0 = relu_1_data_out_0;

endmodule
    