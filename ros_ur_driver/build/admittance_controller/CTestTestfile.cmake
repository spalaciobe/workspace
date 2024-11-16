# CMake generated Testfile for 
# Source directory: /home/sebas/workspace/ros_ur_driver/src/ros2_controllers/admittance_controller
# Build directory: /home/sebas/workspace/ros_ur_driver/build/admittance_controller
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_load_admittance_controller "/usr/bin/python3" "-u" "/opt/ros/humble/share/ament_cmake_test/cmake/run_test.py" "/home/sebas/workspace/ros_ur_driver/build/admittance_controller/test_results/admittance_controller/test_load_admittance_controller.gtest.xml" "--package-name" "admittance_controller" "--output-file" "/home/sebas/workspace/ros_ur_driver/build/admittance_controller/test_results/admittance_controller/test_load_admittance_controller.txt" "--command" "/home/sebas/workspace/ros_ur_driver/build/admittance_controller/test_load_admittance_controller" "--ros-args" "--params-file" "/home/sebas/workspace/ros_ur_driver/src/ros2_controllers/admittance_controller/test/test_params.yaml" "--" "--gtest_output=xml:/home/sebas/workspace/ros_ur_driver/build/admittance_controller/test_results/admittance_controller/test_load_admittance_controller.gtest.xml")
set_tests_properties(test_load_admittance_controller PROPERTIES  TIMEOUT "60" WORKING_DIRECTORY "/home/sebas/workspace/ros_ur_driver/build/admittance_controller" _BACKTRACE_TRIPLES "/opt/ros/humble/share/ament_cmake_test/cmake/ament_add_test.cmake;125;add_test;/opt/ros/humble/share/generate_parameter_library/cmake/generate_parameter_library.cmake;161;ament_add_test;/home/sebas/workspace/ros_ur_driver/src/ros2_controllers/admittance_controller/CMakeLists.txt;70;add_rostest_with_parameters_gmock;/home/sebas/workspace/ros_ur_driver/src/ros2_controllers/admittance_controller/CMakeLists.txt;0;")
add_test(test_admittance_controller "/usr/bin/python3" "-u" "/opt/ros/humble/share/ament_cmake_test/cmake/run_test.py" "/home/sebas/workspace/ros_ur_driver/build/admittance_controller/test_results/admittance_controller/test_admittance_controller.gtest.xml" "--package-name" "admittance_controller" "--output-file" "/home/sebas/workspace/ros_ur_driver/build/admittance_controller/test_results/admittance_controller/test_admittance_controller.txt" "--command" "/home/sebas/workspace/ros_ur_driver/build/admittance_controller/test_admittance_controller" "--ros-args" "--params-file" "/home/sebas/workspace/ros_ur_driver/src/ros2_controllers/admittance_controller/test/test_params.yaml" "--" "--gtest_output=xml:/home/sebas/workspace/ros_ur_driver/build/admittance_controller/test_results/admittance_controller/test_admittance_controller.gtest.xml")
set_tests_properties(test_admittance_controller PROPERTIES  TIMEOUT "60" WORKING_DIRECTORY "/home/sebas/workspace/ros_ur_driver/build/admittance_controller" _BACKTRACE_TRIPLES "/opt/ros/humble/share/ament_cmake_test/cmake/ament_add_test.cmake;125;add_test;/opt/ros/humble/share/generate_parameter_library/cmake/generate_parameter_library.cmake;161;ament_add_test;/home/sebas/workspace/ros_ur_driver/src/ros2_controllers/admittance_controller/CMakeLists.txt;81;add_rostest_with_parameters_gmock;/home/sebas/workspace/ros_ur_driver/src/ros2_controllers/admittance_controller/CMakeLists.txt;0;")
subdirs("gmock")
subdirs("gtest")
