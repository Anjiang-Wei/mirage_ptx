#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000227_ptr, half_t* __restrict__ dtensor10000228_ptr, half_t const* __restrict__ dtensor10000224_ptr, half_t const* __restrict__ dtensor10000225_ptr, half_t const* __restrict__ dtensor10000226_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000985_ptr = (half_t*)(buf + 6272);
  half_t *stensor20000983_ptr = (half_t*)(buf + 128);
  half_t *stensor20000981_ptr = (half_t*)(buf + 71808);
  half_t *stensor30000978_ptr = (half_t*)(buf + 55424);
  half_t *stensor20000978_ptr = (half_t*)(buf + 39040);
  half_t *stensor20000979_ptr = (half_t*)(buf + 22656);
  half_t *stensor30000979_ptr = (half_t*)(buf + 6272);
  half_t *stensor20000982_ptr = (half_t*)(buf + 2176);
  half_t *stensor20000977_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000224 -> stensor 20000977
  const half_t *dtensor10000224_tile_ptr = dtensor10000224_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000224TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000977InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000224TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000225 -> stensor 20000978
  const half_t *dtensor10000225_tile_ptr = dtensor10000225_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000225TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20000978InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000225TileLayout, NUM_THREADS>;
  half_t *stensor20000978_async_copy_buf = stensor30000978_ptr;
  // Copy for G->S: dtensor 10000226 -> stensor 20000979
  const half_t *dtensor10000226_tile_ptr = dtensor10000226_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000226TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20000979InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000226TileLayout, NUM_THREADS>;
  half_t *stensor20000979_async_copy_buf = stensor30000979_ptr;
  
  STensor20000977InputAtom::run(stensor20000977_ptr, dtensor10000224_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000985 -> dtensor 10000228
  half_t *dtensor10000228_tile_ptr = dtensor10000228_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*1 + blockIdx.z*16*512;
  using DTensor10000228TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20000985OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000228TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20000983 -> dtensor 10000227
  half_t *dtensor10000227_tile_ptr = dtensor10000227_ptr  + blockIdx.x*1*2048 + blockIdx.y*1*256 + blockIdx.z*16*1;
  using DTensor10000227TileLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<256>, Int<2048>>>;
  using STensor20000983OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000227TileLayout, Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000982_ptr, thread_idx);
  
  
  using Matmul20000981LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000981LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000981LayoutC = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000981LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000981LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000981Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000981LayoutA, Matmul20000981LayoutB, Matmul20000981LayoutC, Matmul20000981LayoutAAligned, Matmul20000981LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20000985LayoutA = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000985LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000985LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000985LayoutAAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000985LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000985Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000985LayoutA, Matmul20000985LayoutB, Matmul20000985LayoutC, Matmul20000985LayoutAAligned, Matmul20000985LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000985_accum = Matmul20000985Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20000979InputAtom::run(stensor20000979_async_copy_buf, dtensor10000226_tile_ptr, thread_idx);
    STensor20000978InputAtom::run(stensor20000978_async_copy_buf, dtensor10000225_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20000979InputAtom::run(stensor20000979_ptr, dtensor10000226_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20000978InputAtom::run(stensor20000978_ptr, dtensor10000225_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000979_ptr, stensor20000979_async_copy_buf);
      SWAP(stensor20000978_ptr, stensor20000978_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20000981Kernel::get_mma_rC(thread_idx);
      Matmul20000981Kernel::run(mma_rC, stensor20000977_ptr, stensor20000978_ptr, (char*)(buf+0), thread_idx);
      Matmul20000981Kernel::write_back_mma_rC(stensor20000981_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<128>, Int<1>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20000982_ptr, stensor20000981_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20000985Kernel::run(matmul_20000985_accum, stensor20000981_ptr, stensor20000979_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000985Kernel::write_back_mma_rC(stensor20000985_ptr, matmul_20000985_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000983_ptr, stensor20000982_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20000985OutputAtom::run(dtensor10000228_tile_ptr, stensor20000985_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000983OutputAtom::run(dtensor10000227_tile_ptr, stensor20000983_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000229_ptr, half_t const* __restrict__ dtensor10000227_ptr, half_t const* __restrict__ dtensor10000228_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000999_ptr = (half_t*)(buf + 1168);
  half_t *stensor20000998_ptr = (half_t*)(buf + 144);
  half_t *stensor20000996_ptr = (half_t*)(buf + 128);
  half_t *stensor20000997_ptr = (half_t*)(buf + 4480);
  half_t *stensor20000995_ptr = (half_t*)(buf + 4352);
  half_t *stensor20000994_ptr = (half_t*)(buf + 256);
  half_t *stensor20000993_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000227 -> stensor 20000993
  const half_t *dtensor10000227_tile_ptr = dtensor10000227_ptr  + blockIdx.x*1*2048 + blockIdx.y*4*1;
  using DTensor10000227TileLayout = Layout<Shape<Int<4>, Int<8>, Int<1>>, Stride<Int<1>, Int<256>, Int<2048>>>;
  using STensor20000993InputAtom = tb::InputNonChunkedSyncCopy<half_t, Layout<Shape<Int<4>, Int<8>, Int<1>>, Stride<Int<1>, Int<8>, Int<64>>>, DTensor10000227TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000228 -> stensor 20000994
  const half_t *dtensor10000228_tile_ptr = dtensor10000228_ptr  + blockIdx.x*1*131072 + blockIdx.y*4*512;
  using DTensor10000228TileLayout = Layout<Shape<Int<512>, Int<4>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20000994InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<512>, Int<4>, Int<1>>, Stride<Int<1>, Int<512>, Int<2048>>>, DTensor10000228TileLayout, NUM_THREADS>;
  
  STensor20000993InputAtom::run(stensor20000993_ptr, dtensor10000227_tile_ptr, thread_idx);
  STensor20000994InputAtom::run(stensor20000994_ptr, dtensor10000228_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000999 -> dtensor 10000229
  half_t *dtensor10000229_tile_ptr = dtensor10000229_ptr  + blockIdx.x*1*16384 + blockIdx.y*4*64;
  using DTensor10000229TileLayout = Layout<Shape<Int<64>, Int<4>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000999OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000229TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<4>, Int<1>>, Stride<Int<1>, Int<64>, Int<256>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 64, NUM_THREADS>::run(stensor20000995_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000997_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<4>, Int<8>, Int<1>>, Stride<Int<1>, Int<8>, Int<64>>>, Layout<Shape<Int<4>, Int<8>, Int<1>>, Stride<Int<1>, Int<8>, Int<64>>>, NUM_THREADS>;
      Kernel::run(stensor20000995_ptr, stensor20000993_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, decltype(composition(Swizzle<5, 1, 8>{}, Layout<Shape<Int<512>, Int<4>, Int<1>>, Stride<Int<1>, Int<512>, Int<2048>>>{})), Layout<Shape<Int<512>, Int<4>, Int<1>>, Stride<Int<1>, Int<512>, Int<2048>>>, NUM_THREADS>;
      Kernel::run(stensor20000997_ptr, stensor20000994_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<4>, Int<8>, Int<1>>, Stride<Int<1>, Int<8>, Int<64>>>;
    using OutLayout = Layout<Shape<Int<4>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000996_ptr, stensor20000995_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = decltype(composition(Swizzle<5, 1, 8>{}, Layout<Shape<Int<4>, Int<512>, Int<1>>, Stride<Int<512>, Int<1>, Int<2048>>>{}));
    using OutLayout = Layout<Shape<Int<4>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000998_ptr, stensor20000997_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<4>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using In1Layout = Layout<Shape<Int<4>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<4>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<256>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000999_ptr, stensor20000998_ptr, stensor20000996_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000999OutputAtom::run(dtensor10000229_tile_ptr, stensor20000999_ptr, thread_idx);
  }
}


static void _init() {
}


static void _execute_mugraph(std::vector<void const *> input_tensors, std::vector<void*> output_tensors, void* buf, cudaStream_t stream, void * profiler_buffer){
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000227 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000228 = (half_t*)((char*)buf + 8192);
    half_t *dtensor10000224 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000225 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000226 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000227, dtensor10000228, dtensor10000224, dtensor10000225, dtensor10000226);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000229 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000227 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000228 = (half_t*)((char*)buf + 8192);
    dim3 grid_dim(2, 64, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 8576;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 8576);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000229, dtensor10000227, dtensor10000228);
  }
  {
    // OP type: kn_output_op
  }
}


#include <Python.h>
#include <cuda_runtime.h>

static PyObject *launch(PyObject *self, PyObject *args) {
  PyObject *input_list, *output_list, *py_buffer, *py_stream, *py_profiler_buffer;
  void *buffer;
  std::vector<void const *> input_tensors;
  std::vector<void*> output_tensors;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "OOOOO", &input_list, &output_list, &py_buffer, &py_stream, &py_profiler_buffer)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }

  if(!PyList_Check(input_list) || !PyList_Check(output_list)) {
    PyErr_SetString(PyExc_TypeError, "Both arg1 and arg2 must be lists.");
    return NULL;
  }

  Py_ssize_t input_size = PyList_Size(input_list);
  Py_ssize_t output_size = PyList_Size(output_list);

  for(Py_ssize_t i = 0; i < input_size; i++) {
    PyObject *item = PyList_GetItem(input_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (input) to void pointer", i);
      return NULL;
    }
    input_tensors.push_back(PyLong_AsVoidPtr(item));
  }

  for(Py_ssize_t i = 0; i < output_size; i++) {
    PyObject *item = PyList_GetItem(output_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (output) to void pointer", i);
      return NULL;
    }
    output_tensors.push_back(PyLong_AsVoidPtr(item));
  }

  buffer = PyLong_AsVoidPtr(py_buffer);
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);
  cudaStream_t stream = (cudaStream_t)PyLong_AsVoidPtr(py_stream);
  execute_mugraph(input_tensors, output_tensors, buffer, stream, profiler_buffer);

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__mirage_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit___mirage_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
