cmake_minimum_required(VERSION 3.5)
project(get_picture)

# 需要找到 Python3 和 ROS 2 的支持包
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# 如果有 C/C++ 代码
# find_package(ament_cmake_ros REQUIRED)
# find_package(rclcpp REQUIRED)

# 自动收集目录下所有 .py 文件
file(GLOB PYTHON_SCRIPTS
  get_picture_node/*.py
)

# 安装 Python 脚本
install(PROGRAMS
  ${PYTHON_SCRIPTS}
  DESTINATION lib/${PROJECT_NAME}
)

# 其他 C/C++ 节点的构建逻辑
# add_executable(my_c_node src/my_c_node.cpp)
# ament_target_dependencies(my_c_node rclcpp)


ament_package()
