file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(detect STATIC ${sources})
set_target_properties(detect PROPERTIES
        LINKER_LANGUAGE CXX
        )
target_include_directories(detect PUBLIC include)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(glog REQUIRED)
#find_package(gflags REQUIRED)

target_link_libraries(detect PUBLIC
        fmt::fmt
#        gflags
        glog::glog
        ${OpenCV_LIBS}
        )
