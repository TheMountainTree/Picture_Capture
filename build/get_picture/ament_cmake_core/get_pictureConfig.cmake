# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_get_picture_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED get_picture_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(get_picture_FOUND FALSE)
  elseif(NOT get_picture_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(get_picture_FOUND FALSE)
  endif()
  return()
endif()
set(_get_picture_CONFIG_INCLUDED TRUE)

# output package information
if(NOT get_picture_FIND_QUIETLY)
  message(STATUS "Found get_picture: 0.0.1 (${get_picture_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'get_picture' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${get_picture_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(get_picture_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${get_picture_DIR}/${_extra}")
endforeach()
