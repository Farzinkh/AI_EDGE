#include <stdio.h>
#include <stdlib.h>


#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_cli.h"
#include "sd_card.h"


void inference_main(void) {
  mount_sdcard();
  mount_flash();
  esp_cli_init();
  esp_cli_register_cmds();
  vTaskDelay(portMAX_DELAY);
}

void app_main(void)
{
	xTaskCreate((TaskFunction_t)&inference_main, "inference_main", 4 * 1024, NULL, 8, NULL);
  vTaskDelete(NULL);
}