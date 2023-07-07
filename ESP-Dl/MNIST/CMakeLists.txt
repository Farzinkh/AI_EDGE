cmake_minimum_required(VERSION 3.5)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

set(EXTRA_COMPONENT_DIRS "../../components")

set(EXCLUDE_COMPONENTS esp32-camera tflite-lib esp-nn wifi_provisioning mqtt esp_wifi esp_websocket_client esp_http_server esp_https_server esp_http_client esp-tls tcp_transport tcpip_adapter json openssl esp_wifi esp_netif esp_https_ota lwip mbedtls mdns)

project(MNIST)