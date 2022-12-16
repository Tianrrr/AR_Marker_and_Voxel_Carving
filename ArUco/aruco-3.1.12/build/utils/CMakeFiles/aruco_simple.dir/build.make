# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/build

# Include any dependencies generated for this target.
include utils/CMakeFiles/aruco_simple.dir/depend.make

# Include the progress variables for this target.
include utils/CMakeFiles/aruco_simple.dir/progress.make

# Include the compile flags for this target's objects.
include utils/CMakeFiles/aruco_simple.dir/flags.make

utils/CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o: utils/CMakeFiles/aruco_simple.dir/flags.make
utils/CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o: ../utils/aruco_simple.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object utils/CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o"
	cd /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/build/utils && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o -c /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/utils/aruco_simple.cpp

utils/CMakeFiles/aruco_simple.dir/aruco_simple.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aruco_simple.dir/aruco_simple.cpp.i"
	cd /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/build/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/utils/aruco_simple.cpp > CMakeFiles/aruco_simple.dir/aruco_simple.cpp.i

utils/CMakeFiles/aruco_simple.dir/aruco_simple.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aruco_simple.dir/aruco_simple.cpp.s"
	cd /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/build/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/utils/aruco_simple.cpp -o CMakeFiles/aruco_simple.dir/aruco_simple.cpp.s

# Object files for target aruco_simple
aruco_simple_OBJECTS = \
"CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o"

# External object files for target aruco_simple
aruco_simple_EXTERNAL_OBJECTS =

utils/aruco_simple: utils/CMakeFiles/aruco_simple.dir/aruco_simple.cpp.o
utils/aruco_simple: utils/CMakeFiles/aruco_simple.dir/build.make
utils/aruco_simple: src/libaruco.so.3.1.12
utils/aruco_simple: /usr/local/lib/libopencv_calib3d.so.4.6.0
utils/aruco_simple: /usr/local/lib/libopencv_highgui.so.4.6.0
utils/aruco_simple: /usr/local/lib/libopencv_features2d.so.4.6.0
utils/aruco_simple: /usr/local/lib/libopencv_flann.so.4.6.0
utils/aruco_simple: /usr/local/lib/libopencv_videoio.so.4.6.0
utils/aruco_simple: /usr/local/lib/libopencv_imgcodecs.so.4.6.0
utils/aruco_simple: /usr/local/lib/libopencv_imgproc.so.4.6.0
utils/aruco_simple: /usr/local/lib/libopencv_core.so.4.6.0
utils/aruco_simple: utils/CMakeFiles/aruco_simple.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable aruco_simple"
	cd /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/build/utils && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/aruco_simple.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
utils/CMakeFiles/aruco_simple.dir/build: utils/aruco_simple

.PHONY : utils/CMakeFiles/aruco_simple.dir/build

utils/CMakeFiles/aruco_simple.dir/clean:
	cd /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/build/utils && $(CMAKE_COMMAND) -P CMakeFiles/aruco_simple.dir/cmake_clean.cmake
.PHONY : utils/CMakeFiles/aruco_simple.dir/clean

utils/CMakeFiles/aruco_simple.dir/depend:
	cd /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12 /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/utils /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/build /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/build/utils /home/yinghanhuang/TUM/AR/ArUco/aruco-3.1.12/build/utils/CMakeFiles/aruco_simple.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : utils/CMakeFiles/aruco_simple.dir/depend

