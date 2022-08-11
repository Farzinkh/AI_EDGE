/* SD card and FAT filesystem example.
   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

// This example uses SPI peripheral to communicate with SD card.

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include <dirent.h>
#include "esp_log.h"
#include "sd_card.h"

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

void mount_sdcard(sdmmc_card_t ** Card,sdmmc_host_t * host)
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
    ret = spi_bus_initialize(host->slot, &bus_cfg, SPI_DMA_CHAN);
    if (ret != ESP_OK) {
        ESP_LOGE(SDTAG, "Failed to initialize bus.");
        return;
    }

    // This initializes the slot without card detect (CD) and write protect (WP) signals.
    // Modify slot_config.gpio_cd and slot_config.gpio_wp if your board has these signals.
    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = PIN_NUM_CS;
    slot_config.host_id = host->slot;
    ESP_LOGI(SDTAG, "Mounting filesystem");
    sdmmc_card_t  *card;
    ret = esp_vfs_fat_sdspi_mount(mount_point, host, &slot_config, &mount_config, &card);

    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(SDTAG, "Failed to mount filesystem. "
                     "If you want the card to be formatted, set the EXAMPLE_FORMAT_IF_MOUNT_FAILED menuconfig option.");
        } else {
            ESP_LOGE(SDTAG, "Failed to initialize the card (%s). "
                     "Make sure SD card lines have pull-up resistors in place.", esp_err_to_name(ret));
        }
        return ;
    }
    ESP_LOGI(SDTAG, "Filesystem mounted");

    // Card has been initialized, print its properties
    sdmmc_card_print_info(stdout, card);
    *Card=card;
    return ;
}

void unmount_sdcard(sdmmc_card_t ** Card,sdmmc_host_t * host)
{
    // All done, unmount partition and disable SPI peripheral
    sdmmc_card_t  *card;
    card=*Card;
    esp_vfs_fat_sdcard_unmount(mount_point, card);
    ESP_LOGI(SDTAG, "Card unmounted");

    //deinitialize the bus after all devices are removed
    spi_bus_free(host->slot);
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

char * read_image(const char *address)
{
    //char address[]=MOUNT_POINT"/01000033.jpg"
    ESP_LOGI(SDTAG, "Reading image %s", address);
    FILE *f = fopen(address, "rb");
    if (f == NULL) {
        ESP_LOGE(SDTAG, "Failed to open file for reading");
        return NULL;
    }
    fseek(f,0,SEEK_END);
    long fsize=ftell(f);
    fseek(f,0,SEEK_SET);
    char *buffer=malloc(fsize+1);
    fread(buffer,fsize,1,f);
    fclose(f);
    return buffer;
}