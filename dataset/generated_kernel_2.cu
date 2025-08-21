#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000191_ptr, half_t* __restrict__ dtensor10000192_ptr, half_t const* __restrict__ dtensor10000188_ptr, half_t const* __restrict__ dtensor10000189_ptr, half_t const* __restrict__ dtensor10000190_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000803_ptr = (half_t*)(buf + 128);
  half_t *stensor20000800_ptr = (half_t*)(buf + 71808);
  half_t *stensor30000798_ptr = (half_t*)(buf + 55424);
  half_t *stensor20000798_ptr = (half_t*)(buf + 39040);
  half_t *stensor30000797_ptr = (half_t*)(buf + 22656);
  half_t *stensor20000797_ptr = (half_t*)(buf + 6272);
  half_t *stensor20000801_ptr = (half_t*)(buf + 2176);
  half_t *stensor20000796_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000188 -> stensor 20000796
  const half_t *dtensor10000188_tile_ptr = dtensor10000188_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000188TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000796InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000188TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000189 -> stensor 20000797
  const half_t *dtensor10000189_tile_ptr = dtensor10000189_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000189TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20000797InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000189TileLayout, NUM_THREADS>;
  half_t *stensor20000797_async_copy_buf = stensor30000797_ptr;
  // Copy for G->S: dtensor 10000190 -> stensor 20000798
  const half_t *dtensor10000190_tile_ptr = dtensor10000190_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000190TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20000798InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000190TileLayout, NUM_THREADS>;
  half_t *stensor20000798_async_copy_buf = stensor30000798_ptr;
  
  STensor20000796InputAtom::run(stensor20000796_ptr, dtensor10000188_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000801 -> dtensor 10000191
  half_t *dtensor10000191_tile_ptr = dtensor10000191_ptr  + blockIdx.x*1*262144 + blockIdx.y*128*1 + blockIdx.z*16*1024;
  using DTensor10000191TileLayout = Layout<Shape<Int<128>, Int<16>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20000801OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000191TileLayout, Layout<Shape<Int<128>, Int<16>, Int<1>>, Stride<Int<1>, Int<128>, Int<2048>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20000803 -> dtensor 10000192
  half_t *dtensor10000192_tile_ptr = dtensor10000192_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*1 + blockIdx.z*16*512;
  using DTensor10000192TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20000803OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000192TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000801_ptr, thread_idx);
  
  
  using Matmul20000800LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000800LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000800LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20000800LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000800LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000800Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000800LayoutA, Matmul20000800LayoutB, Matmul20000800LayoutC, Matmul20000800LayoutAAligned, Matmul20000800LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20000803LayoutA = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20000803LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000803LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000803LayoutAAligned = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20000803LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000803Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000803LayoutA, Matmul20000803LayoutB, Matmul20000803LayoutC, Matmul20000803LayoutAAligned, Matmul20000803LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000803_accum = Matmul20000803Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20000798InputAtom::run(stensor20000798_async_copy_buf, dtensor10000190_tile_ptr, thread_idx);
    STensor20000797InputAtom::run(stensor20000797_async_copy_buf, dtensor10000189_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20000798InputAtom::run(stensor20000798_ptr, dtensor10000190_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20000797InputAtom::run(stensor20000797_ptr, dtensor10000189_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000798_ptr, stensor20000798_async_copy_buf);
      SWAP(stensor20000797_ptr, stensor20000797_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20000800Kernel::get_mma_rC(thread_idx);
      Matmul20000800Kernel::run(mma_rC, stensor20000796_ptr, stensor20000797_ptr, (char*)(buf+0), thread_idx);
      Matmul20000800Kernel::write_back_mma_rC(stensor20000800_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<128>, Int<16>, Int<1>>, Stride<Int<1>, Int<128>, Int<2048>>>, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>, Int<1>>, Stride<Int<16>, Int<1>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20000801_ptr, stensor20000800_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20000803Kernel::run(matmul_20000803_accum, stensor20000800_ptr, stensor20000798_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000803Kernel::write_back_mma_rC(stensor20000803_ptr, matmul_20000803_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000801OutputAtom::run(dtensor10000191_tile_ptr, stensor20000801_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20000803OutputAtom::run(dtensor10000192_tile_ptr, stensor20000803_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000193_ptr, half_t const* __restrict__ dtensor10000191_ptr, half_t const* __restrict__ dtensor10000192_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000817_ptr = (half_t*)(buf + 656);
  half_t *stensor20000816_ptr = (half_t*)(buf + 144);
  half_t *stensor20000814_ptr = (half_t*)(buf + 128);
  half_t *stensor20000815_ptr = (half_t*)(buf + 20608);
  half_t *stensor20000813_ptr = (half_t*)(buf + 12416);
  half_t *stensor20000812_ptr = (half_t*)(buf + 8320);
  half_t *stensor20000811_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000191 -> stensor 20000811
  const half_t *dtensor10000191_tile_ptr = dtensor10000191_ptr  + blockIdx.x*1*262144 + blockIdx.y*4*1024;
  using DTensor10000191TileLayout = Layout<Shape<Int<1024>, Int<4>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20000811InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<1024>, Int<4>, Int<1>>, Stride<Int<1>, Int<1024>, Int<4096>>>, DTensor10000191TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000192 -> stensor 20000812
  const half_t *dtensor10000192_tile_ptr = dtensor10000192_ptr  + blockIdx.x*1*131072 + blockIdx.y*4*512;
  using DTensor10000192TileLayout = Layout<Shape<Int<512>, Int<4>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20000812InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<512>, Int<4>, Int<1>>, Stride<Int<1>, Int<512>, Int<2048>>>, DTensor10000192TileLayout, NUM_THREADS>;
  
  STensor20000811InputAtom::run(stensor20000811_ptr, dtensor10000191_tile_ptr, thread_idx);
  STensor20000812InputAtom::run(stensor20000812_ptr, dtensor10000192_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000817 -> dtensor 10000193
  half_t *dtensor10000193_tile_ptr = dtensor10000193_ptr  + blockIdx.x*1*16384 + blockIdx.y*4*64;
  using DTensor10000193TileLayout = Layout<Shape<Int<64>, Int<4>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000817OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000193TileLayout, Layout<Shape<Int<64>, Int<4>, Int<1>>, Stride<Int<1>, Int<64>, Int<256>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20000813_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000815_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<1>, Int<1024>, Int<4>>, Stride<Int<1>, Int<1>, Int<1024>>>, Layout<Shape<Int<1>, Int<1024>, Int<4>>, Stride<Int<4096>, Int<1>, Int<1024>>>, NUM_THREADS>;
      Kernel::run(stensor20000813_ptr, stensor20000811_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<512>, Int<4>, Int<1>>, Stride<Int<1>, Int<512>, Int<1>>>, Layout<Shape<Int<512>, Int<4>, Int<1>>, Stride<Int<1>, Int<512>, Int<2048>>>, NUM_THREADS>;
      Kernel::run(stensor20000815_ptr, stensor20000812_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<1>, Int<1024>, Int<4>>, Stride<Int<1>, Int<1>, Int<1024>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<1>, Int<4>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000814_ptr, stensor20000813_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<1>, Int<512>, Int<4>>, Stride<Int<1>, Int<1>, Int<512>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<4>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000816_ptr, stensor20000815_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<1>, Int<64>, Int<4>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<4>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<4>>, Stride<Int<256>, Int<1>, Int<64>>>;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000817_ptr, stensor20000816_ptr, stensor20000814_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000817OutputAtom::run(dtensor10000193_tile_ptr, stensor20000817_ptr, thread_idx);
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
    half_t *dtensor10000191 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000192 = (half_t*)((char*)buf + 1048576);
    half_t *dtensor10000188 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000189 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000190 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000191, dtensor10000192, dtensor10000188, dtensor10000189, dtensor10000190);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000193 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000191 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000192 = (half_t*)((char*)buf + 1048576);
    dim3 grid_dim(2, 64, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 24704;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 24704);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000193, dtensor10000191, dtensor10000192);
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
