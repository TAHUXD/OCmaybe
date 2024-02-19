cmake_minimum_required(VERSION 3.5)
project(cv_gps_system)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED) # Assuming you use sensor_msgs; change as necessary
# Add other packages here as required

# Uncomment if the package has a setup.py
# ament_python_install_package(${PROJECT_NAME})

# Include directories
include_directories(
  include
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  # Add other include directories as required
)

# Declare a C++ library
# add_library(${PROJECT_NAME}_lib
#   src/cv_gps_system.cpp
# )

# ament_target_dependencies(${PROJECT_NAME}_lib
#   "rclcpp"
#   "std_msgs"
#   "sensor_msgs"
#   # Add other dependencies as required
# )

# Declare a C++ executable
# add_executable(${PROJECT_NAME}_node src/cv_gps_system_node.cpp)
# target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME}_lib)
# ament_target_dependencies(${PROJECT_NAME}_node
#   "rclcpp"
#   "std_msgs"
#   "sensor_msgs"
#   # Add other dependencies as required
# )

# Install the executable
# install(TARGETS
#   ${PROJECT_NAME}_node
#   DESTINATION lib/${PROJECT_NAME}
# )

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install configuration files
# install(DIRECTORY
#   config
#   DESTINATION share/${PROJECT_NAME}
# )

# If you have other directories like 'param' or 'rviz', you can also install them in a similar manner

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_lib)
ament_export_dependencies(rclcpp std_msgs sensor_msgs) # Add other dependencies as required

ament_package()