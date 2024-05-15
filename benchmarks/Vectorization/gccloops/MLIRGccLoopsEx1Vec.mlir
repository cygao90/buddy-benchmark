#map = affine_map<(d0) -> (d0 ceildiv STEP_PLACEHOLDER)>
func.func @mlir_gccloopsex1vecSTEP_PLACEHOLDER(%A: memref<?xi32>, %B: memref<?xi32>,
                       %C: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %step = arith.constant STEP_PLACEHOLDER : index
  %c0_i32 = arith.constant 0 : i32
  %n = memref.dim %B, %c0 : memref<?xi32>
  %pass_thru = vector.broadcast %c0_i32 : i32 to vector<STEP_PLACEHOLDERxi32>

  affine.for %idx = 0 to #map(%n) {
    %cur = arith.muli %idx, %step : index
    %tail_len = arith.subi %n, %cur : index
    %tail_flag = arith.cmpi sge, %tail_len, %step : index
    
    scf.if %tail_flag {
      %b_vec = affine.vector_load %B[%idx * STEP_PLACEHOLDER] : memref<?xi32>, vector<STEP_PLACEHOLDERxi32>
      %c_vec = affine.vector_load %C[%idx * STEP_PLACEHOLDER] : memref<?xi32>, vector<STEP_PLACEHOLDERxi32>
      %result_vec = arith.addi %b_vec, %c_vec : vector<STEP_PLACEHOLDERxi32>
      affine.vector_store %result_vec, %A[%idx * STEP_PLACEHOLDER] : memref<?xi32>, vector<STEP_PLACEHOLDERxi32>
    } else {
      %mask_vec = vector.create_mask %tail_len : vector<STEP_PLACEHOLDERxi1>
      %idx_tail = arith.muli %idx, %step : index
      %b_vec_tail = vector.maskedload %B[%idx_tail], %mask_vec, %pass_thru : memref<?xi32>, vector<STEP_PLACEHOLDERxi1>, vector<STEP_PLACEHOLDERxi32> into vector<STEP_PLACEHOLDERxi32>
      %c_vec_tail = vector.maskedload %C[%idx_tail], %mask_vec, %pass_thru : memref<?xi32>, vector<STEP_PLACEHOLDERxi1>, vector<STEP_PLACEHOLDERxi32> into vector<STEP_PLACEHOLDERxi32>
      %result_vec_tail = arith.addi %b_vec_tail, %c_vec_tail : vector<STEP_PLACEHOLDERxi32>
      vector.maskedstore %A[%idx_tail], %mask_vec, %result_vec_tail : memref<?xi32>, vector<STEP_PLACEHOLDERxi1>, vector<STEP_PLACEHOLDERxi32>
    }
  }
  return
}