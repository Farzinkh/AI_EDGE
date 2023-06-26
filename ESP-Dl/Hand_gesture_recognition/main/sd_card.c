/* SD card and FAT filesystem example.
   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

// This example uses SPI peripheral to communicate with SD card.

#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include <dirent.h>
#include "esp_log.h"
#include "sd_card.h"
#include "esp_spiffs.h"

const char mount_point[] = MOUNT_POINT;

// Pin mapping
#if CONFIG_IDF_TARGET_ESP32

#define PIN_NUM_MISO CONFIG_MISO
#define PIN_NUM_MOSI CONFIG_MOSI
#define PIN_NUM_CLK  CONFIG_CLK
#define PIN_NUM_CS   CONFIG_CS

#elif CONFIG_IDF_TARGET_ESP32S2

// adapted for internal test board ESP-32-S3-USB-OTG-Ev-BOARD_V1.0 (with ESP32-S2-MINI-1 module)
#define PIN_NUM_MISO 37
#define PIN_NUM_MOSI 35
#define PIN_NUM_CLK  36
#define PIN_NUM_CS   34

#elif CONFIG_IDF_TARGET_ESP32C3
#define PIN_NUM_MISO 6
#define PIN_NUM_MOSI 4
#define PIN_NUM_CLK  5
#define PIN_NUM_CS   1

#endif //CONFIG_IDF_TARGET_ESP32 || CONFIG_IDF_TARGET_ESP32S2

#if CONFIG_IDF_TARGET_ESP32S2
#define SPI_DMA_CHAN    host.slot
#elif CONFIG_IDF_TARGET_ESP32C3
#define SPI_DMA_CHAN    SPI_DMA_CH_AUTO
#else
#define SPI_DMA_CHAN    1
#endif

sdmmc_host_t host = SDSPI_HOST_DEFAULT();

void write_in_file(const char * tag,const char * text,bool writetosdcard)
{
    // Use POSIX and C standard library functions to work with files.
    // sprintf(text, "%s %s %d  %d", weekday, month, day, year );
    size_t len = strlen(text);
    len=len+strlen(tag)+10;
    char text2write[len];
    time_t rawtime;
    struct tm *info;
    time(&rawtime);
    if (rawtime!=0){
        rawtime = rawtime + (4.5*60*60);
    }
    info = gmtime(&rawtime );
    sprintf(text2write, "%2d:%02d:%02d %s: %s",(info->tm_hour)%24, info->tm_min,info->tm_sec,tag, text);

    // First create a file.
    if (writetosdcard){
        const char *file = MOUNT_POINT"/report.txt";
        ESP_LOGD(SDTAG, "Opening file %s for writing.", file);
        struct stat st;
        FILE *f;
        if (stat(file, &st) == 0) {
            f = fopen(file, "a");
        }else{
            f = fopen(file, "w");
        }
        if (f == NULL) {
            ESP_LOGE(SDTAG, "Failed to open file for writing");
            return;
        }
        fprintf(f, "%s\n", text2write);
        fclose(f);
        ESP_LOGD(SDTAG, "File written");
    } else {
        const char *file = FLASH_MOUNT_POINT"/report.txt";
        ESP_LOGD(FTAG, "Opening file %s for writing.", file);
        struct stat st;
        FILE *f;
        if (stat(file, &st) == 0) {
            f = fopen(file, "a");
        }else{
            f = fopen(file, "w");
        }
            if (f == NULL) {
        ESP_LOGE(FTAG, "Failed to open file for writing");
        return;
        }
        fprintf(f, "%s\n", text2write);
        fclose(f);
        ESP_LOGD(FTAG, "File written");
    }

}
void move_file(const char * src,const char * des)
{
    char    *text2write;
    long    numbytes;
    ESP_LOGI(FTAG, "Opening file %s for reading.", src);
    FILE *f;
    f = fopen(src, "r");
    if (f == NULL) {
        ESP_LOGE(FTAG, "Failed to open file for reading");
    }
    fseek(f, 0L, SEEK_END);
    numbytes = ftell(f);
    fseek(f, 0L, SEEK_SET);	

    text2write = (char*)calloc(numbytes, sizeof(char));	
    fread(text2write, numbytes, 1, f);
    fclose(f);
    ESP_LOGI(SDTAG, "Opening file %s for writing.", des);
    FILE *f2;
    f2 = fopen(des, "w");
    if (f2 == NULL) {
        ESP_LOGE(SDTAG, "Failed to open file for writing");
    }
    fprintf(f2, "%s", text2write);
    fclose(f2);
    if (remove(src) == 0) {
        ESP_LOGI(FTAG,"The file is deleted successfully.");
    } else {
        ESP_LOGE(FTAG,"The file is not deleted.");
    }
}
#ifdef CONFIG_MOUNT_FLASH
void mount_flash()
{
    ESP_LOGI(FTAG, "Initializing SPIFFS");

    esp_vfs_spiffs_conf_t conf = {
      .base_path = FLASH_MOUNT_POINT,
      .partition_label = NULL,
      .max_files = 5,
      .format_if_mount_failed = true
    };

    // Use settings defined above to initialize and mount SPIFFS filesystem.
    // Note: esp_vfs_spiffs_register is an all-in-one convenience function.
    esp_err_t ret = esp_vfs_spiffs_register(&conf);

    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(FTAG, "Failed to mount or format filesystem");
        } else if (ret == ESP_ERR_NOT_FOUND) {
            ESP_LOGE(FTAG, "Failed to find SPIFFS partition");
        } else {
            ESP_LOGE(FTAG, "Failed to initialize SPIFFS (%s)", esp_err_to_name(ret));
        }
        return;
    }

    size_t total = 0, used = 0;
    ret = esp_spiffs_info(conf.partition_label, &total, &used);
    if (ret != ESP_OK) {
        ESP_LOGE(FTAG, "Failed to get SPIFFS partition information (%s). Formatting...", esp_err_to_name(ret));
        esp_spiffs_format(conf.partition_label);
        return;
    } else {
        ESP_LOGI(FTAG, "Partition size: total: %d, used: %d", total, used);
    }
}
#endif

void mount_sdcard()
{
    esp_err_t ret;

    // Options for mounting the filesystem.
    // If format_if_mount_failed is set to true, SD card will be partitioned and
    // formatted in case when mounting fails.
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
#ifdef CONFIG_FORMAT_IF_MOUNT_FAILED
        .format_if_mount_failed = true,
#else
        .format_if_mount_failed = false,
#endif // EXAMPLE_FORMAT_IF_MOUNT_FAILED
        .max_files = 5,
        .allocation_unit_size = 16 * 1024
    };
    ESP_LOGI(SDTAG, "Initializing SD card");

    // Use settings defined above to initialize SD card and mount FAT filesystem.
    // Note: esp_vfs_fat_sdmmc/sdspi_mount is all-in-one convenience functions.
    // Please check its source code and implement error recovery when developing
    // production applications.
    ESP_LOGI(SDTAG, "Using SPI peripheral");

    spi_bus_config_t bus_cfg = {
        .mosi_io_num = PIN_NUM_MOSI,
        .miso_io_num = PIN_NUM_MISO,
        .sclk_io_num = PIN_NUM_CLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4000,
    };
    ret = spi_bus_initialize(host.slot, &bus_cfg, SPI_DMA_CHAN);
    if (ret != ESP_OK) {
        ESP_LOGE(SDTAG, "Failed to initialize bus.");
        esp_restart();
    }

    // This initializes the slot without card detect (CD) and write protect (WP) signals.
    // Modify slot_config.gpio_cd and slot_config.gpio_wp if your board has these signals.
    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = PIN_NUM_CS;
    slot_config.host_id = host.slot;
    ESP_LOGI(SDTAG, "Mounting filesystem");
    sdmmc_card_t  *card;
    ret = esp_vfs_fat_sdspi_mount(mount_point, &host, &slot_config, &mount_config, &card);

    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(SDTAG, "Failed to mount filesystem. "
                     "If you want the card to be formatted, set the EXAMPLE_FORMAT_IF_MOUNT_FAILED menuconfig option.");
        } else {
            ESP_LOGE(SDTAG, "Failed to initialize the card (%s). "
                     "Make sure SD card lines have pull-up resistors in place or sdcard is inserted.", esp_err_to_name(ret));
        }
        //esp_restart();
        spi_bus_free(host.slot);
        return;
    } else{
    ESP_LOGI(SDTAG, "Filesystem mounted");

    // Card has been initialized, print its properties
    sdmmc_card_print_info(stdout, card);
    return ;
    }
}

void unmount_sdcard()
{
    // All done, unmount partition and disable SPI peripheral
    esp_vfs_fat_sdcard_unmount(mount_point, card);
    ESP_LOGI(SDTAG, "Card unmounted");

    //deinitialize the bus after all devices are removed
    spi_bus_free(host.slot);
}

int search_in_sdcard(void)
{
    DIR *d;
    struct dirent *dir;
    d = opendir(MOUNT_POINT);
    ESP_LOGI(SDTAG, "============Listing files in sdcard============");
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            ESP_LOGI(SDTAG, "%s", dir->d_name);
        }
        closedir(d);
    }
    return (0);
}

int search_in_spiffs(const char * name)
{
    DIR *d;
    struct dirent *dir;
    d = opendir(name);
    ESP_LOGI(SDTAG, "============Listing files in %s============",name);
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            ESP_LOGI(SDTAG, "%s", dir->d_name);
        }
        closedir(d);
    }
    return (0);
}

void read_image(IMAGE_t *img,uint8_t dest)
{
    char FileName[40];
    if(dest==1){
        sprintf(FileName, "%s/%s", MOUNT_POINT,img->fname);
    } else {
        sprintf(FileName, "%s/%s", FLASH_MOUNT_POINT,img->fname);
    }
    struct stat st;
	if (stat(FileName, &st) != 0) {
		ESP_LOGE(SDTAG,"[%s] not found",FileName);
	}
    img->fsize=st.st_size;
	ESP_LOGI(SDTAG,"%s size=%ld", FileName, img->fsize);
	// Allocate image memory
	//unsigned char*	image_buffer = NULL;
	size_t image_buffer_len = st.st_size;
	img->buffer = malloc(image_buffer_len);
	if (img->buffer == NULL) {
		ESP_LOGE(SDTAG,"malloc fail. image_buffer_len %d", image_buffer_len);
	}
	// Read image file
	FILE * fp_image = fopen(FileName,"rb");
	if (fp_image == NULL) {
		ESP_LOGE(SDTAG,"[%s] fopen fail.", FileName);
		free(img->buffer);
	}
	for (int i=0;i<st.st_size;i++) {
		fread(&img->buffer[i], sizeof(char), 1, fp_image);
	}
	fclose(fp_image);
}

void write_image(IMAGE_t *img,uint8_t dest)
{
    char FileName[40];
    if(dest==1){
        sprintf(FileName, "%s/%s", MOUNT_POINT,img->fname);
    } else {
        sprintf(FileName, "%s/%s", FLASH_MOUNT_POINT,img->fname);
    }
    FILE *f = fopen(FileName, "wb");
    if (f == NULL)
    {
        ESP_LOGE(SDTAG, "Failed to open file for writing");
        ESP_LOGE(SDTAG, "[%s]",FileName);
        return;
    }
    fwrite(img->buffer, 1,img->fsize, f);
    fclose(f);
    ESP_LOGI(SDTAG, "File written");
    free(img->buffer);
}