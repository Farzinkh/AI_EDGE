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

class HANDRECOGNATION : public Model<int16_t>
{
private:
	Reshape<int16_t> l1;
	Conv2D<int16_t> l2;
	Relu<int16_t> l3;
	MaxPool2D<int16_t> l4;
	Transpose<int16_t> l5;
	Reshape<int16_t> l6;
	Conv2D<int16_t> l7;
	Relu<int16_t> l8;
	Conv2D<int16_t> l9;

public:
	Softmax<int16_t> l10;

	HANDRECOGNATION () :
				l1(Reshape<int16_t>({96,96,1},"l1_reshape")),
				l2(Conv2D<int16_t>(-8, get_sequential_2_conv2d_2_biasadd_filter(), get_sequential_2_conv2d_2_biasadd_bias(), get_sequential_2_conv2d_2_biasadd_activation(), PADDING_VALID, {}, 1, 1, "l2")),
				l3(Relu<int16_t>("l3_relu")),
				l4(MaxPool2D<int16_t>({5,5},PADDING_VALID,{}, 5, 5, "l4")),
				l5(Transpose<int16_t>({2,1,0},"l5_transpose")),
				l6(Reshape<int16_t>({1,1,2592},"l6_reshape")),
				l7(Conv2D<int16_t>(-9, get_fused_gemm_0_filter(), get_fused_gemm_0_bias(), get_fused_gemm_0_activation(), PADDING_VALID, {}, 1, 1, "l7")),
				l8(Relu<int16_t>("l8_relu")),
				l9(Conv2D<int16_t>(-9, get_fused_gemm_1_filter(), get_fused_gemm_1_bias(), NULL, PADDING_VALID, {}, 1, 1, "l9")),
				l10(Softmax<int16_t>(-18,"l10")){}
	void build(Tensor<int16_t> &input)
	{
		this->l1.build(input,true);
		this->l2.build(this->l1.get_output(),true);
		this->l3.build(this->l2.get_output(),true);
		this->l4.build(this->l3.get_output(),true);
		this->l5.build(this->l4.get_output(),true);
		this->l6.build(this->l5.get_output(),true);
		this->l7.build(this->l6.get_output(),true);
		this->l8.build(this->l7.get_output(),true);
		this->l9.build(this->l8.get_output(),true);
		this->l10.build(this->l9.get_output(),true);
	}
	void call(Tensor<int16_t> &input)
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

		this->l7.call(this->l6.get_output());
		this->l6.get_output().free_element();

		this->l8.call(this->l7.get_output());
		this->l7.get_output().free_element();

		this->l9.call(this->l8.get_output());
		this->l8.get_output().free_element();

		this->l10.call(this->l9.get_output());
		this->l9.get_output().free_element();
	}

};
