idf_build_get_property(target IDF_TARGET)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(srcs  main.c ../model/handrecognation_coefficient.cpp inference.cpp benchmark.c)
set(src_dirs  ../model)
set(include_dirs  ../model ./include $ENV{ESP_DL_PATH}/include $ENV{ESP_DL_PATH}/include/tool $ENV{ESP_DL_PATH}/include/typedef $ENV{ESP_DL_PATH}/include/nn $ENV{ESP_DL_PATH}/include/layer $ENV{ESP_DL_PATH}/include/math)

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