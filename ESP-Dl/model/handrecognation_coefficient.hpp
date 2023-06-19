#pragma once

#include <stdint.h>
#include "dl_constant.hpp"

namespace handrecognation_coefficient
{
    const dl::Filter<int16_t> *get_sequential_2_conv2d_2_biasadd_filter();
    const dl::Bias<int16_t> *get_sequential_2_conv2d_2_biasadd_bias();
    const dl::Activation<int16_t> *get_sequential_2_conv2d_2_biasadd_activation();
    const dl::Filter<int16_t> *get_fused_gemm_0_filter();
    const dl::Bias<int16_t> *get_fused_gemm_0_bias();
    const dl::Activation<int16_t> *get_fused_gemm_0_activation();
    const dl::Filter<int16_t> *get_fused_gemm_1_filter();
    const dl::Bias<int16_t> *get_fused_gemm_1_bias();
}
