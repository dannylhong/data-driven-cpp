function(add_example basename)

  # add the exectuable and link it to the Python libs
  add_executable(${basename} ${basename}.cpp)
  target_link_libraries(${basename} -lpython3.10)

endfunction(add_example)

# add the executables
add_example(CH01_SEC02)