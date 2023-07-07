#ifndef SD_CARD
#define SD_CARD

#include "sdmmc_cmd.h"
#include "esp_vfs_fat.h"
#include <sys/stat.h>
#include <sys/unistd.h>
#include <string.h>

typedef struct {
    char fname[30];
    long fsize;
    unsigned char *buffer;
} IMAGE_t;

static const char *SDTAG = "SD";
static const char *FTAG = "Flash";


#define MOUNT_POINT "/sdcard" 
#define FLASH_MOUNT_POINT "/spiffs"

void write_in_file(const char * tag,const char * text,bool writetosdcard);
void move_file(const char * src,const char * des);
void mount_sdcard();
void unmount_sdcard();
int search_in_spiffs(const char * name);
int search_in_sdcard(void);
void read_image(IMAGE_t *img,uint8_t dest);
void write_image(IMAGE_t *img,uint8_t dest);
void mount_flash();

#endif