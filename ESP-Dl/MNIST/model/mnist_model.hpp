#pragma once
#include "dl_layer_model.hpp"
#include "dl_layer_relu.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_max_pool2d.hpp"
#include "dl_layer_softmax.hpp"
#include "dl_layer_transpose.hpp"
#include "mnist_coefficient.hpp"
#include <stdint.h>

using namespace dl;
using namespace layer;
using namespace mnist_coefficient;

class MNIST : public Model<int8_t>
{
private:
	Reshape<int8_t> l1;
	Conv2D<int8_t> l2;
	Conv2D<int8_t> l3;

public:
	Conv2D<int8_t> l4;

	MNIST () :
				l1(Reshape<int8_t>({1,1,784},"l1_reshape")),
				l2(Conv2D<int8_t>(5, get_fused_gemm_0_filter(), get_fused_gemm_0_bias(), get_fused_gemm_0_activation(), PADDING_VALID, {}, 1, 1, "l2")),
				l3(Conv2D<int8_t>(5, get_fused_gemm_1_filter(), get_fused_gemm_1_bias(), NULL, PADDING_VALID, {}, 1, 1, "l3")),
				l4(Conv2D<int8_t>(7, get_fused_gemm_2_filter(), get_fused_gemm_2_bias(), NULL, PADDING_VALID, {}, 1, 1, "l4")){}
				void build(Tensor<int8_t> &input)
	{
		this->l1.build(input,true);
		this->l2.build(this->l1.get_output(),true);
		this->l3.build(this->l2.get_output(),true);
		this->l4.build(this->l3.get_output(),true);
	}
	void call(Tensor<int8_t> &input)
	{
		this->l1.call(input);
		input.free_element();

		this->l2.call(this->l1.get_output());
		this->l1.get_output().free_element();

		this->l3.call(this->l2.get_output());
		this->l2.get_output().free_element();

		this->l4.call(this->l3.get_output());
		this->l3.get_output().free_element();
	}

};
