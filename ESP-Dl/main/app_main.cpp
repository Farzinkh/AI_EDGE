#include <stdio.h>
#include <stdlib.h>

#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "dl_tool.hpp"
#include "mnist_model.hpp"


extern "C" void app_main(void)
{
	int input_height = 28;
	int input_width = 28;
	int input_channel = 1;
	int input_exponent = -7;
	int8_t *model_input = (int8_t *)dl::tool::malloc_aligned_prefer(input_height*input_width*input_channel, sizeof(int8_t *));
	for(int i=0 ;i<input_height*input_width*input_channel; i++){
		float normalized_input = test_image[i] / 255.0; //normalization
		model_input[i] = (int8_t)DL_CLIP(normalized_input * (1 << -input_exponent), -128, 127);
	}
	Tensor<int8_t> input;
	input.set_element((int8_t *)model_input).set_exponent(input_exponent).set_shape({input_height, input_width, input_channel}).set_auto_free(false);

	MNIST model;
	dl::tool::Latency latency;
	latency.start();
	model.forward(input);
	latency.end();
	latency.print("MNIST", "forward");

	//parse
	int8_t *score = model.l4.get_output().get_element_ptr();
	int8_t max_score = score[0];
	int max_index = 0;
	printf("%d, ", max_score);

	for (size_t i = 1; i < 10; i++)
	{
		printf("%d, ", score[i]);
		if (score[i] > max_score)
		{
			max_score = score[i];
			max_index = i;
		}
	}
	printf("Prediction Result: %d", max_index);
}