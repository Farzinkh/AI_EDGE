#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <esp_log.h>
#include <esp_console.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <driver/uart.h>

#include "esp_main.h"
#include "benchmark.h"
#include "esp_timer.h"
#include "sd_card.h"
#include <dirent.h>

#include <sys/stat.h>

static int stop;
static const char *TAG = "CLI";

static int cpu_dump_cli_handler(int argc, char *argv[])
{
    /* Just to go to the next line */
    printf("\n");
#ifndef CONFIG_FREERTOS_GENERATE_RUN_TIME_STATS
    printf("%s: To use this utility enable: Component config --> FreeRTOS --> Enable FreeRTOS to collect run time stats\n", TAG);
#else
    char *buf = calloc(1, 2 * 1024);
    vTaskGetRunTimeStats(buf);
    printf("%s: Run Time Stats:\n%s\n", TAG, buf);
    free(buf);
#endif
    return 0;
}

static int mem_dump_cli_handler(int argc, char *argv[])
{
    /* Just to go to the next line */
    const float MB=0.000000954;
    printf("\n");
    printf("\tDescription\tInternal\t\tSPIRAM\n");
    printf("Current Free Memory\t%.6f\t\t%.6f\n",
           (heap_caps_get_free_size(MALLOC_CAP_8BIT) - heap_caps_get_free_size(MALLOC_CAP_SPIRAM))*MB,
           heap_caps_get_free_size(MALLOC_CAP_SPIRAM)*MB);
    printf("Largest Free Block\t%.6f\t\t%.6f\n",
           heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL)*MB,
           heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM)*MB);
    printf("Min. Ever Free Size\t%.6f\t\t%.6f\n",
           heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL)*MB,
           heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM)*MB);
    return 0;
}
static int inference_benchmark_handler(int argc,char *argv[])
{
    printf("\n");
    ESP_LOGI(SDTAG, "============Running benchmark============");
    DIR *d;
    int detect=0;
    int type=0;
    int counter=CONFIG_INSTANCE_NUMBER;
    int verbose=100;
    int v=verbose;
    char partialtext[81];
    //char text[sizeof(partialtext)*verbose];
    char * text;
    text= (char*)malloc(sizeof(partialtext)*(verbose+7));
    memset(text,0,strlen(text));
    int nStatus;
    struct stat sFileStatus;
    char* pBuffer="";
    int  nBufferSize=0;
    FILE* f=NULL;
    bool bufferlocated=false;
    const TickType_t xDelay = 10 / portTICK_PERIOD_MS;
    memset(&sFileStatus, 0x0, sizeof(struct stat));
    uint32_t read_image_time=0;
    uint32_t detect_time=0;
    uint32_t write_time=0;
    char * address;
    address= (char*)malloc(265);
    //unsigned char buffer[784];
    struct dirent *dir;
    if (remove(MOUNT_POINT"/REPORT.TXT") == 0) {
        ESP_LOGI(SDTAG,"Previous Report file deleted successfully.");
    } 
    if (remove(FLASH_MOUNT_POINT"/report.txt") == 0) {
        ESP_LOGI(FTAG,"Previous Report file deleted successfully.");
    } 
    d = opendir(MOUNT_POINT);
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            read_image_time = esp_timer_get_time();
            if (!strcmp(dir->d_name,"TRASH-~1") || !strcmp(dir->d_name,"ANDROI~1") || !strcmp(dir->d_name,"SYSTEM~1") || !strcmp(dir->d_name,"REPORT.TXT"))
            {
                continue;
            }
            strcpy(address,MOUNT_POINT"/");
            strcat(address,dir->d_name);
            ESP_LOGI(SDTAG, "%s", address);
            // Status the file size.
            if (0 != (nStatus = stat(address, &sFileStatus)))
            {
                ESP_LOGE(SDTAG,"stat() failed on file: '%s.  Error Code: %d\n",
                        dir->d_name, nStatus);
                continue;
            }
            else if(!bufferlocated){
                if(0 == (pBuffer = (char*)malloc(nBufferSize = sFileStatus.st_size)))
                {
                    ESP_LOGE(SDTAG, "buffer allocation failed.\n");
                    continue;
                }
                bufferlocated=true;
            }
            if(0 == (f = fopen(address, "rb")))
            {
                ESP_LOGE(SDTAG, "Failed to open file for reading");
                continue;
            }
            else if(nBufferSize != (nStatus = fread(pBuffer,1,nBufferSize,f)))
            {
                ESP_LOGE(SDTAG, "fread() failed to read size reported by stat() %d.\n",nBufferSize);
                continue;
            }
            fclose(f);
            read_image_time = (esp_timer_get_time() - read_image_time)/1000;
            detect_time = esp_timer_get_time();
            detect=run_inference((void *) pBuffer);
            type=dir->d_name[strlen(dir->d_name)-5]-'0'; // -1 if raw file -5 if .jpg
            detect_time = (esp_timer_get_time() - detect_time)/1000;
            write_time = esp_timer_get_time();
            if (detect==type)
            {
                sprintf(partialtext,"%s Reading image delay is %hu ms Inference delay is %hu ms :: %d True",address,read_image_time, detect_time,detect);
                
            } else {
                sprintf(partialtext,"%s Reading image delay is %hu ms Inference delay is %hu ms :: %d False",address,read_image_time, detect_time,detect);
            }
            v-=1;
            if (v==0){
                strcat(text,partialtext);
                write_in_file(TAG,text,false);
                memset(text,0,strlen(text));
                v=verbose;
                write_time = (esp_timer_get_time() - write_time )/1000;
                ESP_LOGI(SDTAG,"%hu ms %hu ms",read_image_time,write_time);
            } else {
                strcat(text,partialtext);
                strcat(text,"\n");
                ESP_LOGI(SDTAG,"%hu ms",read_image_time);
            }
            if (counter==0){
                break;
            } else {
                counter -=1;
            }
            vTaskDelay(xDelay);
            free(pBuffer);
            bufferlocated=false;
        }
        closedir(d);
    }
    const char *src = FLASH_MOUNT_POINT"/report.txt";
    const char *des = MOUNT_POINT"/report.txt";
    move_file(src,des);
    ESP_LOGI(SDTAG, "============Benchmark Finnished============");
    return 0;
}
static int move_result(int argc,char *argv[])
{
    const char *src = FLASH_MOUNT_POINT"/report.txt";
    const char *des = MOUNT_POINT"/report.txt";
    move_file(src,des);
    return 0;
}

static esp_console_cmd_t diag_cmds[] = {
    {
        .command = "mem-dump",
        .help = "",
        .func = mem_dump_cli_handler,
    },
    {
        .command = "cpu-dump",
        .help = "",
        .func = cpu_dump_cli_handler,
    },
    {
        .command = "move_result",
        .help = "move report.txt file to SD card.",
        .func = move_result,
    },
    {
        .command = "run_benchmark",
        .help = "detect all images from sdcard.",
        .func = inference_benchmark_handler,
    },
};

static void esp_cli_task(void *arg)
{
    int uart_num = (int) arg;
    uint8_t linebuf[256];
    int i, cmd_ret;
    esp_err_t ret;
    QueueHandle_t uart_queue;
    uart_event_t event;

    ESP_LOGI(TAG, "Initialising UART on port %d", uart_num);
    uart_driver_install(uart_num, 256, 0, 8, &uart_queue, 0);
    /* Initialize the console */
    esp_console_config_t console_config = {
            .max_cmdline_args = 8,
            .max_cmdline_length = 256,
    };

    esp_console_init(&console_config);
    esp_console_register_help_command();
    while (!stop) {
        uart_write_bytes(uart_num, "\n>> ", 4);
        memset(linebuf, 0, sizeof(linebuf));
        i = 0;
        do {
            ret = xQueueReceive(uart_queue, (void * )&event, (portTickType)portMAX_DELAY);
            if (ret != pdPASS) {
                if(stop == 1) {
                    break;
                } else {
                    continue;
                }
            }
            if (event.type == UART_DATA) {
                while (uart_read_bytes(uart_num, (uint8_t *) &linebuf[i], 1, 0)) {
                    if (linebuf[i] == '\r') {
                        uart_write_bytes(uart_num, "\r\n", 2);
                    } else {
                        uart_write_bytes(uart_num, (char *) &linebuf[i], 1);
                    }
                    i++;
                }
            }
        } while ((i < 255) && linebuf[i-1] != '\r');
        if (stop) {
            break;
        }
        /* Remove the truncating \r\n */
        linebuf[strlen((char *)linebuf) - 1] = '\0';
        ret = esp_console_run((char *) linebuf, &cmd_ret);
        if (ret < 0) {
            printf("%s: Console dispatcher error\n", TAG);
            break;
        }
    }
    ESP_LOGE(TAG, "Stopped CLI");
    vTaskDelete(NULL);
}

int esp_cli_register_cmds()
{
    int cmds_num = sizeof(diag_cmds) / sizeof(esp_console_cmd_t);
    int i;
    for (i = 0; i < cmds_num; i++) {
        ESP_LOGI(TAG, "Registering command: %s", diag_cmds[i].command);
        esp_console_cmd_register(&diag_cmds[i]);
    }
    return 0;
}

int esp_cli_init()
{
    static int cli_started;
    if (cli_started) {
        return 0;
    }
#define ESP_CLI_STACK_SIZE (15 * 1024)
    //StackType_t *task_stack = (StackType_t *) calloc(1, ESP_CLI_STACK_SIZE);
    //static StaticTask_t task_buf;
    if(pdPASS != xTaskCreate(&esp_cli_task, "cli_task", ESP_CLI_STACK_SIZE, NULL, tskIDLE_PRIORITY,NULL)) {
        ESP_LOGE(TAG, "Couldn't create task");
        return -1;
    }
    cli_started = 1;
    return 0;
}
