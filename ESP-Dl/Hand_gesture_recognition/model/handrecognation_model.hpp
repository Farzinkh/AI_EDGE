#pragma once
#include "dl_layer_model.hpp"
#include "dl_layer_relu.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_max_pool2d.hpp"
#include "dl_layer_softmax.hpp"
#include "dl_layer_transpose.hpp"
#include "handrecognation_coefficient.hpp"
#include <stdint.h>

using namespace dl;
using namespace layer;
using namespace handrecognation_coefficient;

class HANDRECOGNATION : public Model<int8_t>
{
private:
	Conv2D<int8_t> l1;
	MaxPool2D<int8_t> l2;
	Reshape<int8_t> l3;
	Conv2D<int8_t> l4;
	Conv2D<int8_t> l5;

public:
	Softmax<int8_t> l6;

	HANDRECOGNATION () :
				l1(Conv2D<int8_t>(1, get_statefulpartitionedcall_sequential_conv2d_biasadd_filter(), get_statefulpartitionedcall_sequential_conv2d_biasadd_bias(), get_statefulpartitionedcall_sequential_conv2d_biasadd_activation(), PADDING_VALID, {}, 1, 1, "l1")),
				l2(MaxPool2D<int8_t>({5,5},PADDING_VALID,{}, 5, 5, "l2")),
				l3(Reshape<int8_t>({1,1,2592},"l3_reshape")),
				l4(Conv2D<int8_t>(2, get_fused_gemm_0_filter(), get_fused_gemm_0_bias(), get_fused_gemm_0_activation(), PADDING_VALID, {}, 1, 1, "l4")),
				l5(Conv2D<int8_t>(1, get_fused_gemm_1_filter(), get_fused_gemm_1_bias(), NULL, PADDING_VALID, {}, 1, 1, "l5")),
				l6(Softmax<int8_t>(-6,"l6")){}
	void build(Tensor<int8_t> &input)
	{
		this->l1.build(input,true);
		this->l2.build(this->l1.get_output(),true);
		this->l3.build(this->l2.get_output(),true);
		this->l4.build(this->l3.get_output(),true);
		this->l5.build(this->l4.get_output(),true);
		this->l6.build(this->l5.get_output(),true);
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

		this->l5.call(this->l4.get_output());
		this->l4.get_output().free_element();

		this->l6.call(this->l5.get_output());
		this->l5.get_output().free_element();
	}

};
