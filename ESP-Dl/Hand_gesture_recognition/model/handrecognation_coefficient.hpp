#pragma once

#include <stdint.h>
#include "dl_constant.hpp"

namespace handrecognation_coefficient
{
    const dl::Filter<int8_t> *get_statefulpartitionedcall_sequential_conv2d_biasadd_filter();
    const dl::Bias<int8_t> *get_statefulpartitionedcall_sequential_conv2d_biasadd_bias();
    const dl::Activation<int8_t> *get_statefulpartitionedcall_sequential_conv2d_biasadd_activation();
    const dl::Filter<int8_t> *get_fused_gemm_0_filter();
    const dl::Bias<int8_t> *get_fused_gemm_0_bias();
    const dl::Activation<int8_t> *get_fused_gemm_0_activation();
    const dl::Filter<int8_t> *get_fused_gemm_1_filter();
    const dl::Bias<int8_t> *get_fused_gemm_1_bias();
}
