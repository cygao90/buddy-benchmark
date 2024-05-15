//===- MLIRGccLoopsEx1Benchmark.cpp --------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the benchmark for buddy-opt tool in buddy-mlir project.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>
#include <buddy/Core/Container.h>
#include <iostream>

// Declare the gccloopsex1vec C interface.
extern "C" {
void _mlir_ciface_mlir_gccloopsex1vec16(MemRef<int, 1> *output,
                              MemRef<int, 1> *input1,
                              MemRef<int, 1> *input2);
void _mlir_ciface_mlir_gccloopsex1vec32(MemRef<int, 1> *output,
                              MemRef<int, 1> *input1,
                              MemRef<int, 1> *input2);
void _mlir_ciface_mlir_gccloopsex1vec64(MemRef<int, 1> *output,
                              MemRef<int, 1> *input1,
                              MemRef<int, 1> *input2);
void _mlir_ciface_mlir_gccloopsex1vec128(MemRef<int, 1> *output,
                              MemRef<int, 1> *input1,
                              MemRef<int, 1> *input2);
void _mlir_ciface_mlir_gccloopsex1vec256(MemRef<int, 1> *output,
                              MemRef<int, 1> *input1,
                              MemRef<int, 1> *input2);
}

// Define input and output sizes.
intptr_t sizesInputArrayMLIRGccLoopsEx1Vec_1[1] = {128};
intptr_t sizesInputArrayMLIRGccLoopsEx1Vec_2[1] = {128};
intptr_t sizesOutputArrayMLIRGccLoopsEx1Vec[1] = {128};
// Define the MemRef container for inputs and output.
MemRef<int, 1> inputMLIRGccLoopsEx1Vec_1(sizesInputArrayMLIRGccLoopsEx1Vec_1, 2);
MemRef<int, 1> inputMLIRGccLoopsEx1Vec_2(sizesInputArrayMLIRGccLoopsEx1Vec_2, 3);
MemRef<int, 1> outputMLIRGccLoopsEx1Vec16(sizesOutputArrayMLIRGccLoopsEx1Vec, 0);
MemRef<int, 1> outputMLIRGccLoopsEx1Vec32(sizesOutputArrayMLIRGccLoopsEx1Vec, 0);
MemRef<int, 1> outputMLIRGccLoopsEx1Vec64(sizesOutputArrayMLIRGccLoopsEx1Vec, 0);
MemRef<int, 1> outputMLIRGccLoopsEx1Vec128(sizesOutputArrayMLIRGccLoopsEx1Vec, 0);
MemRef<int, 1> outputMLIRGccLoopsEx1Vec256(sizesOutputArrayMLIRGccLoopsEx1Vec, 0);

#define DEFINE_VECTOR_BENCHMARK(width) \
  static void MLIR_GccLoopsEx1Vec##width(benchmark::State &state) { \
    for (auto _ : state) { \
      for (int i = 0; i < state.range(0); ++i) { \
        _mlir_ciface_mlir_gccloopsex1vec##width(&outputMLIRGccLoopsEx1Vec##width, &inputMLIRGccLoopsEx1Vec_1, \
                                &inputMLIRGccLoopsEx1Vec_2); \
      } \
    } \
  } \
  BENCHMARK(MLIR_GccLoopsEx1Vec##width)->Arg(1); \


DEFINE_VECTOR_BENCHMARK(16)
DEFINE_VECTOR_BENCHMARK(32)
DEFINE_VECTOR_BENCHMARK(64)
DEFINE_VECTOR_BENCHMARK(128)
DEFINE_VECTOR_BENCHMARK(256)

// Generate result image.
void generateResultMLIRGccLoopsEx1Vec() {
  // Define the MemRef descriptor for inputs and output.
  MemRef<int, 1> input1(sizesInputArrayMLIRGccLoopsEx1Vec_1, 2);
  MemRef<int, 1> input2(sizesInputArrayMLIRGccLoopsEx1Vec_2, 3);

  MemRef<int, 1> output16(sizesOutputArrayMLIRGccLoopsEx1Vec, 0);
  MemRef<int, 1> output32(sizesOutputArrayMLIRGccLoopsEx1Vec, 0);
  MemRef<int, 1> output64(sizesOutputArrayMLIRGccLoopsEx1Vec, 0);
  MemRef<int, 1> output128(sizesOutputArrayMLIRGccLoopsEx1Vec, 0);
  MemRef<int, 1> output256(sizesOutputArrayMLIRGccLoopsEx1Vec, 0);
  // Run the gccloopsex1.
  _mlir_ciface_mlir_gccloopsex1vec16(&output16, &input1, &input2);
  _mlir_ciface_mlir_gccloopsex1vec32(&output32, &input1, &input2);
  _mlir_ciface_mlir_gccloopsex1vec64(&output64, &input1, &input2);
  _mlir_ciface_mlir_gccloopsex1vec128(&output128, &input1, &input2);
  _mlir_ciface_mlir_gccloopsex1vec256(&output256, &input1, &input2);
  // Print the output.
  std::cout << "--------------------------------------------------------"
            << std::endl;
#define OUTPUT(width) \
  std::cout << "MLIR_GccLoopsEx1Vec"#width": MLIR GccLoopsEx1Vec Operation" << std::endl; \
  std::cout << "[ "; \
  for (size_t i = 0; i < output##width.getSize(); i++) { \
    std::cout << output##width.getData()[i] << " "; \
  } \
  std::cout << "]" << std::endl; \

OUTPUT(16)
OUTPUT(32)
OUTPUT(64)
OUTPUT(128)
OUTPUT(256)
}
