#map = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>

func.func @mlir_gccloopsex1dynvec(%output: memref<?xi32>, %input1: memref<?xi32>, %input2: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  // Get the dimension of the workload.
  %dim_size = memref.dim %input1, %c0 : memref<?xi32>
  // Perform dynamic vector addition.
  // Returns four times the physical vl for element type i32.
  %vl = vector_exp.get_vl i32, 4 : index

  scf.for %idx = %c0 to %dim_size step %vl { // Tiling
    %it_vl = affine.min #map(%idx)[%vl, %dim_size]
    vector_exp.set_vl %it_vl : index {
      %vec_input1 = vector.load %input1[%idx] : memref<?xi32>, vector<[1]xi32> // vector<?xi32>
      %vec_input2 = vector.load %input2[%idx] : memref<?xi32>, vector<[1]xi32> // vector<?xi32>
      %vec_output = arith.addi %vec_input1, %vec_input2 : vector<[1]xi32> // vector<?xi32>
      vector.store %vec_output, %output[%idx] : memref<?xi32>, vector<[1]xi32> // vector<?xi32>
      vector.yield
    }
  }
  return
}