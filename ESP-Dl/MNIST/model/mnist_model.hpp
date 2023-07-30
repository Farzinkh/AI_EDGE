#pragma once
#include "dl_layer_model.hpp"
#include "dl_layer_relu.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_max_pool2d.hpp"
#include "dl_layer_softmax.hpp"
#include "dl_layer_transpose.hpp"
#include "dl_layer_fullyconnected.hpp"
#include "mnist_coefficient.hpp"
#include <stdint.h>

using namespace dl;
using namespace layer;
using namespace mnist_coefficient;

class MNIST : public Model<int8_t>
{
private:
	Reshape<int8_t> l1;
	FullyConnected<int8_t> l2;

public:
	FullyConnected<int8_t> l3;

	MNIST () :
				l1(Reshape<int8_t>({1,1,784},"l1_reshape")),
				l2(FullyConnected<int8_t>(6, get_fused_gemm_0_filter(), get_fused_gemm_0_bias(), get_fused_gemm_0_activation(),true, "l2")),
				l3(FullyConnected<int8_t>(8, get_fused_gemm_1_filter(), get_fused_gemm_1_bias(), NULL,true, "l3")){}
				void build(Tensor<int8_t> &input)
	{
		this->l1.build(input,true);
		this->l2.build(this->l1.get_output(),true);
		this->l3.build(this->l2.get_output(),true);
	}
	void call(Tensor<int8_t> &input)
	{
		this->l1.call(input);
		input.free_element();

		this->l2.call(this->l1.get_output());
		this->l1.get_output().free_element();

		this->l3.call(this->l2.get_output());
		this->l2.get_output().free_element();
	}

};
