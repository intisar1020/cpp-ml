add_library(cpp-ml
    msnet_inference.cpp

)

target_include_directories(cpp-ml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(cpp-ml PUBLIC ${ONNXRUNTIME_LIBRARIES})
