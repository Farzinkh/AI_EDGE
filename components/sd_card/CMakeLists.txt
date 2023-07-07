set(COMPONENT_SRCS sd_card.c)
set(COMPONENT_ADD_INCLUDEDIRS "./include")
set(COMPONENT_REQUIRES sdmmc fatfs spiffs)
register_component()
