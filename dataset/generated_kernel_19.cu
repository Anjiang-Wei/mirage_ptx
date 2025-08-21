#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000293_ptr, half_t* __restrict__ dtensor10000294_ptr, half_t const* __restrict__ dtensor10000290_ptr, half_t const* __restrict__ dtensor10000291_ptr, half_t const* __restrict__ dtensor10000292_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001329_ptr = (half_t*)(buf + 128);
  half_t *stensor20001326_ptr = (half_t*)(buf + 49280);
  half_t *stensor30001324_ptr = (half_t*)(buf + 41088);
  half_t *stensor20001324_ptr = (half_t*)(buf + 32896);
  half_t *stensor30001323_ptr = (half_t*)(buf + 24704);
  half_t *stensor20001323_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001327_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001322_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000290 -> stensor 20001322
  const half_t *dtensor10000290_tile_ptr = dtensor10000290_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000290TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001322InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000290TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000291 -> stensor 20001323
  const half_t *dtensor10000291_tile_ptr = dtensor10000291_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*1;
  using DTensor10000291TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001323InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000291TileLayout, NUM_THREADS>;
  half_t *stensor20001323_async_copy_buf = stensor30001323_ptr;
  // Copy for G->S: dtensor 10000292 -> stensor 20001324
  const half_t *dtensor10000292_tile_ptr = dtensor10000292_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*64;
  using DTensor10000292TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001324InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000292TileLayout, NUM_THREADS>;
  half_t *stensor20001324_async_copy_buf = stensor30001324_ptr;
  
  STensor20001322InputAtom::run(stensor20001322_ptr, dtensor10000290_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001327 -> dtensor 10000293
  half_t *dtensor10000293_tile_ptr = dtensor10000293_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*1 + blockIdx.z*64*1024;
  using DTensor10000293TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20001327OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000293TileLayout, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20001329 -> dtensor 10000294
  half_t *dtensor10000294_tile_ptr = dtensor10000294_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*1 + blockIdx.z*64*1024;
  using DTensor10000294TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20001329OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000294TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001327_ptr, thread_idx);
  
  
  using Matmul20001326LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001326LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001326LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001326LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001326LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001326Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001326LayoutA, Matmul20001326LayoutB, Matmul20001326LayoutC, Matmul20001326LayoutAAligned, Matmul20001326LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001329LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001329LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001329LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001329LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001329LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001329Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001329LayoutA, Matmul20001329LayoutB, Matmul20001329LayoutC, Matmul20001329LayoutAAligned, Matmul20001329LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001329_accum = Matmul20001329Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001324InputAtom::run(stensor20001324_async_copy_buf, dtensor10000292_tile_ptr, thread_idx);
    STensor20001323InputAtom::run(stensor20001323_async_copy_buf, dtensor10000291_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001324InputAtom::run(stensor20001324_ptr, dtensor10000292_tile_ptr + 4096*(for_idx+1), thread_idx);
        STensor20001323InputAtom::run(stensor20001323_ptr, dtensor10000291_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001324_ptr, stensor20001324_async_copy_buf);
      SWAP(stensor20001323_ptr, stensor20001323_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001326Kernel::get_mma_rC(thread_idx);
      Matmul20001326Kernel::run(mma_rC, stensor20001322_ptr, stensor20001323_ptr, (char*)(buf+0), thread_idx);
      Matmul20001326Kernel::write_back_mma_rC(stensor20001326_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001327_ptr, stensor20001326_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001329Kernel::run(matmul_20001329_accum, stensor20001326_ptr, stensor20001324_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001329Kernel::write_back_mma_rC(stensor20001329_ptr, matmul_20001329_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001327OutputAtom::run(dtensor10000293_tile_ptr, stensor20001327_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20001329OutputAtom::run(dtensor10000294_tile_ptr, stensor20001329_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000295_ptr, half_t const* __restrict__ dtensor10000293_ptr, half_t const* __restrict__ dtensor10000294_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001343_ptr = (half_t*)(buf + 128);
  half_t *stensor20001342_ptr = (half_t*)(buf + 2192);
  half_t *stensor20001340_ptr = (half_t*)(buf + 2176);
  half_t *stensor30001337_ptr = (half_t*)(buf + 5248);
  half_t *stensor20001337_ptr = (half_t*)(buf + 4224);
  half_t *stensor20001338_ptr = (half_t*)(buf + 3200);
  half_t *stensor30001338_ptr = (half_t*)(buf + 2176);
  half_t *stensor20001341_ptr = (half_t*)(buf + 1152);
  half_t *stensor20001339_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000293 -> stensor 20001337
  const half_t *dtensor10000293_tile_ptr = dtensor10000293_ptr  + blockIdx.x*1*262144 + blockIdx.y*2*1024;
  using DTensor10000293TileLayout = Layout<Shape<Int<256>, Int<2>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20001337InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<256>, Int<2>, Int<1>>, Stride<Int<1>, Int<256>, Int<512>>>, DTensor10000293TileLayout, NUM_THREADS>;
  half_t *stensor20001337_async_copy_buf = stensor30001337_ptr;
  // Copy for G->S: dtensor 10000294 -> stensor 20001338
  const half_t *dtensor10000294_tile_ptr = dtensor10000294_ptr  + blockIdx.x*1*262144 + blockIdx.y*2*1024;
  using DTensor10000294TileLayout = Layout<Shape<Int<256>, Int<2>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20001338InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<256>, Int<2>, Int<1>>, Stride<Int<1>, Int<256>, Int<512>>>, DTensor10000294TileLayout, NUM_THREADS>;
  half_t *stensor20001338_async_copy_buf = stensor30001338_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001343 -> dtensor 10000295
  half_t *dtensor10000295_tile_ptr = dtensor10000295_ptr  + blockIdx.x*1*16384 + blockIdx.y*2*64;
  using DTensor10000295TileLayout = Layout<Shape<Int<64>, Int<2>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001343OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000295TileLayout, Layout<Shape<Int<64>, Int<2>, Int<1>>, Stride<Int<1>, Int<64>, Int<128>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 512, NUM_THREADS>::run(stensor20001339_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 512, NUM_THREADS>::run(stensor20001341_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20001338InputAtom::run(stensor20001338_async_copy_buf, dtensor10000294_tile_ptr, thread_idx);
    STensor20001337InputAtom::run(stensor20001337_async_copy_buf, dtensor10000293_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001338InputAtom::run(stensor20001338_ptr, dtensor10000294_tile_ptr + 256*(for_idx+1), thread_idx);
        STensor20001337InputAtom::run(stensor20001337_ptr, dtensor10000293_tile_ptr + 256*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001338_ptr, stensor20001338_async_copy_buf);
      SWAP(stensor20001337_ptr, stensor20001337_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<256>, Int<2>, Int<1>>, Stride<Int<1>, Int<256>, Int<1>>>, Layout<Shape<Int<256>, Int<2>, Int<1>>, Stride<Int<1>, Int<256>, Int<512>>>, NUM_THREADS>;
      Kernel::run(stensor20001339_ptr, stensor20001337_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<256>, Int<2>, Int<1>>, Stride<Int<1>, Int<256>, Int<512>>>, Layout<Shape<Int<256>, Int<2>, Int<1>>, Stride<Int<1>, Int<256>, Int<512>>>, NUM_THREADS>;
      Kernel::run(stensor20001341_ptr, stensor20001338_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<1>, Int<256>, Int<2>>, Stride<Int<1>, Int<1>, Int<256>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<1>, Int<2>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001340_ptr, stensor20001339_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<1>, Int<256>, Int<2>>, Stride<Int<512>, Int<1>, Int<256>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001342_ptr, stensor20001341_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<2>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<128>, Int<1>, Int<64>>>;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001343_ptr, stensor20001342_ptr, stensor20001340_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001343OutputAtom::run(dtensor10000295_tile_ptr, stensor20001343_ptr, thread_idx);
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
    half_t *dtensor10000293 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000294 = (half_t*)((char*)buf + 1048576);
    half_t *dtensor10000290 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000291 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000292 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 57472;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 57472);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000293, dtensor10000294, dtensor10000290, dtensor10000291, dtensor10000292);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000295 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000293 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000294 = (half_t*)((char*)buf + 1048576);
    dim3 grid_dim(2, 128, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 6272;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 6272);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000295, dtensor10000293, dtensor10000294);
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
