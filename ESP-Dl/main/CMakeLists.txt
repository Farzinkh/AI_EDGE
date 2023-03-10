idf_build_get_property(target IDF_TARGET)

set(srcs  app_main.cpp ../model/mnist_coefficient.cpp)
set(src_dirs  ../model)
set(include_dirs  ../model $ENV{ESP_DL_PATH}/include $ENV{ESP_DL_PATH}/include/tool $ENV{ESP_DL_PATH}/include/typedef $ENV{ESP_DL_PATH}/include/nn $ENV{ESP_DL_PATH}/include/layer)

idf_component_register(SRCS ${srcs} SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs}
REQUIRES ${requires})
set(lib     libdl.a)

if(${IDF_TARGET} STREQUAL "esp32")
	set(links   "-L $ENV{ESP_DL_PATH}/lib/esp32")

elseif(${IDF_TARGET} STREQUAL "esp32s2")
	set(links   "-L $ENV{ESP_DL_PATH}/lib/esp32s2")

elseif(${IDF_TARGET} STREQUAL "esp32s3")
	set(links   "-L $ENV{ESP_DL_PATH}/lib/esp32s3")

endif()

target_link_libraries(${COMPONENT_TARGET} ${links})
target_link_libraries(${COMPONENT_TARGET} ${lib})