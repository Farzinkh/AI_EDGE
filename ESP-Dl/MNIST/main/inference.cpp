#include <stdio.h>
#include <stdlib.h>

#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <esp_log.h>

#include "dl_tool.hpp"
#include "mnist_model.hpp"
#include "esp_main.h"

int input_height = 28;
int input_width = 28;
int input_channel = 1;
int input_exponent = -7;
int size=input_height*input_width*input_channel;
static const char *TAG = "INF";

#ifdef __cplusplus
extern "C" {
#endif

MNIST model;
bool warmed=false;

void warm_up(void *image)
{
	Tensor<int16_t> input;
	input.set_element((int16_t *)image).set_exponent(input_exponent).set_shape({input_height, input_width, input_channel}).set_auto_free(false);
	dl::tool::Latency latency;
	for(int i = 0; i < 5; i++)
	{
		latency.start();
		model.forward(input);
		latency.end();
		latency.print("warming", "forward");
	}
}

int run_inference(void *image)
{
	if(!warmed){
		warm_up(image);
		warmed=true;
	}
	#ifdef CONFIG_PREPROCESS
	int8_t *pointer_to_img;
	pointer_to_img = (int8_t *) image;
	int16_t *model_input = (int16_t *)dl::tool::malloc_aligned_prefer(size, sizeof(int16_t *));
	for(int i=0 ;i<size; i++){
		float normalized_input = pointer_to_img[i] / 255.0; //normalization
		model_input[i] = (int16_t)DL_CLIP(normalized_input * (1 << -input_exponent), -32768, 32767);
	}
	Tensor<int16_t> input;
	input.set_element((int16_t *)model_input).set_exponent(input_exponent).set_shape({input_height, input_width, input_channel}).set_auto_free(false);
	#else
	Tensor<int16_t> input;
	input.set_element((int16_t *)image).set_exponent(input_exponent).set_shape({input_height, input_width, input_channel}).set_auto_free(false);
	#endif

	#ifdef CONFIG_INFERENCE_LOG
	dl::tool::Latency latency;
	latency.start();
	#endif
	model.forward(input);
	#ifdef CONFIG_INFERENCE_LOG
	latency.end();
	latency.print("MNIST", "forward");

	#endif
	#ifdef CONFIG_PREPROCESS
	dl::tool::free_aligned_prefer(model_input);
	#endif

	//parse
	auto *score = model.l4.get_output().get_element_ptr();
	auto max_score = score[0];
	int max_index = 0;

	for (size_t i = 1; i < 10; i++)
	{
		if (score[i] > max_score)
		{
			max_score = score[i];
			max_index = i;
		}
	}
	#ifdef CONFIG_INFERENCE_LOG
	switch (max_index)
	{
	case 0:
		ESP_LOGI(TAG,"0");
		break;
	case 1:
		ESP_LOGI(TAG,"1");
		break;
	case 2:
		ESP_LOGI(TAG,"2");
		break;
	case 3:
		ESP_LOGI(TAG,"3");
		break;
	case 4:
		ESP_LOGI(TAG,"4");
		break;
	case 5:
		ESP_LOGI(TAG,"5");
		break;
	case 6:
		ESP_LOGI(TAG,"6");
		break;
	case 7:
		ESP_LOGI(TAG,"7");
		break;
	case 8:
		ESP_LOGI(TAG,"8");
		break;
	case 9:
		ESP_LOGI(TAG,"9");
		break;
	default:
		ESP_LOGE(TAG,"No result");
	}
	#endif
	return (max_index);
}
#ifdef __cplusplus
}
#endif