add_library(data SHARED 
    DataLoader.h DataLoader.cpp
    DataMap.h DataMap.cpp
    RuntimeDataType.h RuntimeDataType.cpp
    common.h data.h
)
target_link_libraries(data blas)