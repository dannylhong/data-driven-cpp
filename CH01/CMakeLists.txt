function(add_example basename)

  # add the exectuable and link it to the Python libs
  add_executable(${basename} ${basename}.cpp)
  target_link_libraries(${basename} -lpython3.10)
  target_link_libraries(${basename} ${OpenCV_LIBS})
  target_link_libraries(${basename} Matplot++::matplot)
  target_link_libraries(${basename} matioCpp::matioCpp)

endfunction(add_example)

# add the executables
add_example(CH01_SEC02)
add_example(CH01_SEC03_Rotation)
add_example(CH01_SEC04_1_Linear)
add_example(CH01_SEC04_2_Cement)
add_example(CH01_SEC04_3_Housing)
add_example(CH01_SEC05_1_PCAGaussian)
add_example(CH01_SEC05_2_OvarianCancer)
add_example(CH01_SEC06_1)
add_example(CH01_SEC06_2_3_4)