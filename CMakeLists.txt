add_library(vision-lib
    msnet_inference.cpp

)

target_include_directories(vision-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(vision-lib PUBLIC ${ONNXRUNTIME_LIBRARIES})
