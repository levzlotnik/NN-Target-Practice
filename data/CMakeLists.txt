add_library(data SHARED 
    DataLoader.h DataLoader.cpp
    common.h data.h
    DataFrame.h DataFrame.cpp
)
target_link_libraries(data blas)