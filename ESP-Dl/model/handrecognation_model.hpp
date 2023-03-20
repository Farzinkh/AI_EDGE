#pragma once
#include "dl_layer_model.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_max_pool2d.hpp"
#include "dl_layer_softmax.hpp"
#include "handrecognation_coefficient.hpp"
#include <stdint.h>

using namespace dl;
using namespace layer;
using namespace handrecognation_coefficient;

class HANDRECOGNATION : public Model<int8_t>
{
private:
	Reshape<int8_t> l1;
	Conv2D<int8_t> l2;
	MaxPool2D<int8_t> l3;
	Conv2D<int8_t> l4;
	MaxPool2D<int8_t> l5;
	Conv2D<int8_t> l6;
	MaxPool2D<int8_t> l7;
	Conv2D<int8_t> l8;
	MaxPool2D<int8_t> l9;
	Reshape<int8_t> l10;
	Conv2D<int8_t> l11;
	Conv2D<int8_t> l12;

public:
	Softmax<int8_t> l13;

	HANDRECOGNATION () :
				l1(Reshape<int8_t>({1,1,9216},"l1_reshape")),
				l2(Conv2D<int8_t>(0, get_sequential_5_conv2d_12_biasadd_filter(), get_sequential_5_conv2d_12_biasadd_bias(), get_sequential_5_conv2d_12_biasadd_activation(), PADDING_SAME_END, {}, 1, 1, "l2")),
				l3(MaxPool2D<int8_t>({2,2},PADDING_VALID,{}, 2, 2, "l3")),
				l4(Conv2D<int8_t>(0, get_sequential_5_conv2d_13_biasadd_filter(), get_sequential_5_conv2d_13_biasadd_bias(), get_sequential_5_conv2d_13_biasadd_activation(), PADDING_SAME_END, {}, 1, 1, "l4")),
				l5(MaxPool2D<int8_t>({2,2},PADDING_VALID,{}, 2, 2, "l5")),
				l6(Conv2D<int8_t>(0, get_sequential_5_conv2d_14_biasadd_filter(), get_sequential_5_conv2d_14_biasadd_bias(), get_sequential_5_conv2d_14_biasadd_activation(), PADDING_SAME_END, {}, 1, 1, "l6")),
				l7(MaxPool2D<int8_t>({2,2},PADDING_VALID,{}, 2, 2, "l7")),
				l8(Conv2D<int8_t>(-1, get_sequential_5_conv2d_15_biasadd_filter(), get_sequential_5_conv2d_15_biasadd_bias(), get_sequential_5_conv2d_15_biasadd_activation(), PADDING_SAME_END, {}, 1, 1, "l8")),
				l9(MaxPool2D<int8_t>({2,2},PADDING_VALID,{}, 2, 2, "l9")),
				l10(Reshape<int8_t>({1,1,1024},"l10_reshape")),
				l11(Conv2D<int8_t>(-1, get_fused_gemm_0_filter(), get_fused_gemm_0_bias(), get_fused_gemm_0_activation(), PADDING_SAME_END, {}, 1, 1, "l11")),
				l12(Conv2D<int8_t>(-1, get_fused_gemm_1_filter(), get_fused_gemm_1_bias(), NULL, PADDING_SAME_END, {}, 1, 1, "l12")),
				l13(Softmax<int8_t>(-6,"l13")){}
	void build(Tensor<int8_t> &input)
	{
		this->l1.build(input);
		this->l2.build(this->l1.get_output());
		this->l3.build(this->l2.get_output());
		this->l4.build(this->l3.get_output());
		this->l5.build(this->l4.get_output());
		this->l6.build(this->l5.get_output());
		this->l7.build(this->l6.get_output());
		this->l8.build(this->l7.get_output());
		this->l9.build(this->l8.get_output());
		this->l10.build(this->l9.get_output());
		this->l11.build(this->l10.get_output());
		this->l12.build(this->l11.get_output());
		this->l13.build(this->l12.get_output());
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

		this->l7.call(this->l6.get_output());
		this->l6.get_output().free_element();

		this->l8.call(this->l7.get_output());
		this->l7.get_output().free_element();

		this->l9.call(this->l8.get_output());
		this->l8.get_output().free_element();

		this->l10.call(this->l9.get_output());
		this->l9.get_output().free_element();

		this->l11.call(this->l10.get_output());
		this->l10.get_output().free_element();

		this->l12.call(this->l11.get_output());
		this->l11.get_output().free_element();

		this->l13.call(this->l12.get_output());
		this->l12.get_output().free_element();
	}

};
