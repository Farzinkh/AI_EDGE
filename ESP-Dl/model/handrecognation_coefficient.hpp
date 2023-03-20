#pragma once

#include <stdint.h>
#include "dl_constant.hpp"

namespace handrecognation_coefficient
{
    const dl::Filter<int8_t> *get_sequential_5_conv2d_12_biasadd_filter();
    const dl::Bias<int8_t> *get_sequential_5_conv2d_12_biasadd_bias();
    const dl::Activation<int8_t> *get_sequential_5_conv2d_12_biasadd_activation();
    const dl::Filter<int8_t> *get_sequential_5_conv2d_13_biasadd_filter();
    const dl::Bias<int8_t> *get_sequential_5_conv2d_13_biasadd_bias();
    const dl::Activation<int8_t> *get_sequential_5_conv2d_13_biasadd_activation();
    const dl::Filter<int8_t> *get_sequential_5_conv2d_14_biasadd_filter();
    const dl::Bias<int8_t> *get_sequential_5_conv2d_14_biasadd_bias();
    const dl::Activation<int8_t> *get_sequential_5_conv2d_14_biasadd_activation();
    const dl::Filter<int8_t> *get_sequential_5_conv2d_15_biasadd_filter();
    const dl::Bias<int8_t> *get_sequential_5_conv2d_15_biasadd_bias();
    const dl::Activation<int8_t> *get_sequential_5_conv2d_15_biasadd_activation();
    const dl::Filter<int8_t> *get_fused_gemm_0_filter();
    const dl::Bias<int8_t> *get_fused_gemm_0_bias();
    const dl::Activation<int8_t> *get_fused_gemm_0_activation();
    const dl::Filter<int8_t> *get_fused_gemm_1_filter();
    const dl::Bias<int8_t> *get_fused_gemm_1_bias();
}
