**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Zachary Leong
  * [LinkedIn](https://linkedin.com/in/zleong), [personal website](https://zacharyleong.com)
* Tested on: Windows 11, Ultra 9 185H @ 2.30GHz 32GB, RTX 4060 Laptop (personal)

### pull request
- 

### boids



#### cmake lists modification
I moved `include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")` so that the CUDA toolkit is included along with Windows builds as CMake Tools in VS Code doesn't automatically include this path like Visual Studio does.

I also added a compile option for release mode to include the `-lineinfo` tag for NSight Compute performance analysis.

Added: `$<COMPILE_LANGUAGE:CUDA>>:-G>" "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>"`
