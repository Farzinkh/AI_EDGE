#include <stdio.h>
#include <stdlib.h>

#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <esp_log.h>

#include "dl_tool.hpp"
#include "handrecognation_model.hpp"
#include "esp_main.h"

int input_height = 96;
int input_width = 96;
int input_channel = 1;
int input_exponent = 1;
int size=input_height*input_width*input_channel;
static const char *TAG = "INF";

#ifdef __cplusplus
extern "C" {
#endif

int run_inference(void *image)
{
	#ifdef CONFIG_PREPROCESS
	int8_t *pointer_to_img;
	pointer_to_img = (int8_t *) image;
	int8_t *model_input = (int8_t *)dl::tool::malloc_aligned_prefer(size, sizeof(int8_t *));
	for(int i=0 ;i<size; i++){
		static float normalized_input = pointer_to_img[i];
		model_input[i] = (int8_t)DL_CLIP(normalized_input * (1 << -input_exponent), -128, 127);
	}
	Tensor<int8_t> input;
	input.set_element((int8_t *)model_input).set_exponent(input_exponent).set_shape({input_height, input_width, input_channel}).set_auto_free(false);
	#else
	Tensor<int8_t> input;
	input.set_element((int8_t *)image).set_exponent(input_exponent).set_shape({input_height, input_width, input_channel}).set_auto_free(false);
	#endif

	HANDRECOGNATION model;
	#ifdef CONFIG_INFERENCE_LOG
	dl::tool::Latency latency;
	latency.start();
	#endif
	model.forward(input);
	#ifdef CONFIG_INFERENCE_LOG
	latency.end();
	latency.print("HANDRECOGNATION", "forward");

	#endif
	#ifdef CONFIG_PREPROCESS
	dl::tool::free_aligned_prefer(model_input);
	#endif

	//parse
	auto *score = model.l13.get_output().get_element_ptr();
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
		ESP_LOGI(TAG,"Palm")
		break;
	case 1:
		ESP_LOGI(TAG,"I")
		break;
	case 2:
		ESP_LOGI(TAG,"fist")
		break;
	case 3:
		ESP_LOGI(TAG,"fist_moved")
		break;
	case 4:
		ESP_LOGI(TAG,"thumb")
		break;
	case 5:
		ESP_LOGI(TAG,"index")
		break;
	case 6:
		ESP_LOGI(TAG,"ok")
		break;
	case 7:
		ESP_LOGI(TAG,"palm_moved")
		break;
	case 8:
		ESP_LOGI(TAG,"c")
		break;
	case 9:
		ESP_LOGI(TAG,"down")
		break;
	default:
		ESP_LOGE(TAG,"No result")
	}
	#endif
	return (max_index);
}
#ifdef __cplusplus
}
#endif
