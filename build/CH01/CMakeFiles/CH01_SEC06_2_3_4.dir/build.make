# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/daniel/Documents/study/databook/data-driven-cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/daniel/Documents/study/databook/data-driven-cpp/build

# Include any dependencies generated for this target.
include CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/compiler_depend.make

# Include the progress variables for this target.
include CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/progress.make

# Include the compile flags for this target's objects.
include CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/flags.make

CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.o: CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/flags.make
CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.o: ../CH01/CH01_SEC06_2_3_4.cpp
CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.o: CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/daniel/Documents/study/databook/data-driven-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.o"
	cd /home/daniel/Documents/study/databook/data-driven-cpp/build/CH01 && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.o -MF CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.o.d -o CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.o -c /home/daniel/Documents/study/databook/data-driven-cpp/CH01/CH01_SEC06_2_3_4.cpp

CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.i"
	cd /home/daniel/Documents/study/databook/data-driven-cpp/build/CH01 && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/daniel/Documents/study/databook/data-driven-cpp/CH01/CH01_SEC06_2_3_4.cpp > CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.i

CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.s"
	cd /home/daniel/Documents/study/databook/data-driven-cpp/build/CH01 && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/daniel/Documents/study/databook/data-driven-cpp/CH01/CH01_SEC06_2_3_4.cpp -o CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.s

# Object files for target CH01_SEC06_2_3_4
CH01_SEC06_2_3_4_OBJECTS = \
"CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.o"

# External object files for target CH01_SEC06_2_3_4
CH01_SEC06_2_3_4_EXTERNAL_OBJECTS =

CH01/CH01_SEC06_2_3_4: CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/CH01_SEC06_2_3_4.cpp.o
CH01/CH01_SEC06_2_3_4: CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/build.make
CH01/CH01_SEC06_2_3_4: /usr/local/lib/libopencv_highgui.so.4.5.5
CH01/CH01_SEC06_2_3_4: /home/daniel/Libraries/vcpkg/installed/x64-linux/lib/libmatplot.a
CH01/CH01_SEC06_2_3_4: /home/daniel/Libraries/matio-cpp/build/lib/libmatioCpp.so.0.2.0
CH01/CH01_SEC06_2_3_4: /usr/local/lib/libopencv_videoio.so.4.5.5
CH01/CH01_SEC06_2_3_4: /usr/local/lib/libopencv_imgcodecs.so.4.5.5
CH01/CH01_SEC06_2_3_4: /usr/local/lib/libopencv_imgproc.so.4.5.5
CH01/CH01_SEC06_2_3_4: /usr/local/lib/libopencv_core.so.4.5.5
CH01/CH01_SEC06_2_3_4: /usr/local/lib/libopencv_cudev.so.4.5.5
CH01/CH01_SEC06_2_3_4: /home/daniel/Libraries/vcpkg/installed/x64-linux/lib/libnodesoup.a
CH01/CH01_SEC06_2_3_4: /usr/lib/x86_64-linux-gnu/libmatio.so
CH01/CH01_SEC06_2_3_4: CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/daniel/Documents/study/databook/data-driven-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CH01_SEC06_2_3_4"
	cd /home/daniel/Documents/study/databook/data-driven-cpp/build/CH01 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CH01_SEC06_2_3_4.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/build: CH01/CH01_SEC06_2_3_4
.PHONY : CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/build

CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/clean:
	cd /home/daniel/Documents/study/databook/data-driven-cpp/build/CH01 && $(CMAKE_COMMAND) -P CMakeFiles/CH01_SEC06_2_3_4.dir/cmake_clean.cmake
.PHONY : CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/clean

CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/depend:
	cd /home/daniel/Documents/study/databook/data-driven-cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/daniel/Documents/study/databook/data-driven-cpp /home/daniel/Documents/study/databook/data-driven-cpp/CH01 /home/daniel/Documents/study/databook/data-driven-cpp/build /home/daniel/Documents/study/databook/data-driven-cpp/build/CH01 /home/daniel/Documents/study/databook/data-driven-cpp/build/CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CH01/CMakeFiles/CH01_SEC06_2_3_4.dir/depend

