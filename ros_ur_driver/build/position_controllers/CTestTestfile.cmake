# CMake generated Testfile for 
# Source directory: /home/sebas/workspace/ros_ur_driver/src/ros2_controllers/position_controllers
# Build directory: /home/sebas/workspace/ros_ur_driver/build/position_controllers
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_load_joint_group_position_controller "/usr/bin/python3" "-u" "/opt/ros/humble/share/ament_cmake_test/cmake/run_test.py" "/home/sebas/workspace/ros_ur_driver/build/position_controllers/test_results/position_controllers/test_load_joint_group_position_controller.gtest.xml" "--package-name" "position_controllers" "--output-file" "/home/sebas/workspace/ros_ur_driver/build/position_controllers/ament_cmake_gmock/test_load_joint_group_position_controller.txt" "--command" "/home/sebas/workspace/ros_ur_driver/build/position_controllers/test_load_joint_group_position_controller" "--gtest_output=xml:/home/sebas/workspace/ros_ur_driver/build/position_controllers/test_results/position_controllers/test_load_joint_group_position_controller.gtest.xml")
set_tests_properties(test_load_joint_group_position_controller PROPERTIES  LABELS "gmock" REQUIRED_FILES "/home/sebas/workspace/ros_ur_driver/build/position_controllers/test_load_joint_group_position_controller" TIMEOUT "60" WORKING_DIRECTORY "/home/sebas/workspace/ros_ur_driver/build/position_controllers" _BACKTRACE_TRIPLES "/opt/ros/humble/share/ament_cmake_test/cmake/ament_add_test.cmake;125;add_test;/opt/ros/humble/share/ament_cmake_gmock/cmake/ament_add_gmock.cmake;106;ament_add_test;/opt/ros/humble/share/ament_cmake_gmock/cmake/ament_add_gmock.cmake;52;_ament_add_gmock;/home/sebas/workspace/ros_ur_driver/src/ros2_controllers/position_controllers/CMakeLists.txt;40;ament_add_gmock;/home/sebas/workspace/ros_ur_driver/src/ros2_controllers/position_controllers/CMakeLists.txt;0;")
add_test(test_joint_group_position_controller "/usr/bin/python3" "-u" "/opt/ros/humble/share/ament_cmake_test/cmake/run_test.py" "/home/sebas/workspace/ros_ur_driver/build/position_controllers/test_results/position_controllers/test_joint_group_position_controller.gtest.xml" "--package-name" "position_controllers" "--output-file" "/home/sebas/workspace/ros_ur_driver/build/position_controllers/ament_cmake_gmock/test_joint_group_position_controller.txt" "--command" "/home/sebas/workspace/ros_ur_driver/build/position_controllers/test_joint_group_position_controller" "--gtest_output=xml:/home/sebas/workspace/ros_ur_driver/build/position_controllers/test_results/position_controllers/test_joint_group_position_controller.gtest.xml")
set_tests_properties(test_joint_group_position_controller PROPERTIES  LABELS "gmock" REQUIRED_FILES "/home/sebas/workspace/ros_ur_driver/build/position_controllers/test_joint_group_position_controller" TIMEOUT "60" WORKING_DIRECTORY "/home/sebas/workspace/ros_ur_driver/build/position_controllers" _BACKTRACE_TRIPLES "/opt/ros/humble/share/ament_cmake_test/cmake/ament_add_test.cmake;125;add_test;/opt/ros/humble/share/ament_cmake_gmock/cmake/ament_add_gmock.cmake;106;ament_add_test;/opt/ros/humble/share/ament_cmake_gmock/cmake/ament_add_gmock.cmake;52;_ament_add_gmock;/home/sebas/workspace/ros_ur_driver/src/ros2_controllers/position_controllers/CMakeLists.txt;51;ament_add_gmock;/home/sebas/workspace/ros_ur_driver/src/ros2_controllers/position_controllers/CMakeLists.txt;0;")
subdirs("gmock")
subdirs("gtest")
