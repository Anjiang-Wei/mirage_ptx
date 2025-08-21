#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000281_ptr, half_t* __restrict__ dtensor10000282_ptr, half_t const* __restrict__ dtensor10000278_ptr, half_t const* __restrict__ dtensor10000279_ptr, half_t const* __restrict__ dtensor10000280_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001269_ptr = (half_t*)(buf + 128);
  half_t *stensor20001266_ptr = (half_t*)(buf + 49280);
  half_t *stensor30001264_ptr = (half_t*)(buf + 41088);
  half_t *stensor20001264_ptr = (half_t*)(buf + 32896);
  half_t *stensor30001263_ptr = (half_t*)(buf + 24704);
  half_t *stensor20001263_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001267_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001262_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000278 -> stensor 20001262
  const half_t *dtensor10000278_tile_ptr = dtensor10000278_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000278TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001262InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000278TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000279 -> stensor 20001263
  const half_t *dtensor10000279_tile_ptr = dtensor10000279_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*1;
  using DTensor10000279TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001263InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000279TileLayout, NUM_THREADS>;
  half_t *stensor20001263_async_copy_buf = stensor30001263_ptr;
  // Copy for G->S: dtensor 10000280 -> stensor 20001264
  const half_t *dtensor10000280_tile_ptr = dtensor10000280_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*64;
  using DTensor10000280TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001264InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000280TileLayout, NUM_THREADS>;
  half_t *stensor20001264_async_copy_buf = stensor30001264_ptr;
  
  STensor20001262InputAtom::run(stensor20001262_ptr, dtensor10000278_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001267 -> dtensor 10000281
  half_t *dtensor10000281_tile_ptr = dtensor10000281_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*256 + blockIdx.z*64*1;
  using DTensor10000281TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001267OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000281TileLayout, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20001269 -> dtensor 10000282
  half_t *dtensor10000282_tile_ptr = dtensor10000282_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*256 + blockIdx.z*64*1;
  using DTensor10000282TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001269OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000282TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001267_ptr, thread_idx);
  
  
  using Matmul20001266LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001266LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001266LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001266LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001266LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001266Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001266LayoutA, Matmul20001266LayoutB, Matmul20001266LayoutC, Matmul20001266LayoutAAligned, Matmul20001266LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001269LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001269LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001269LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001269LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001269LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001269Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001269LayoutA, Matmul20001269LayoutB, Matmul20001269LayoutC, Matmul20001269LayoutAAligned, Matmul20001269LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001269_accum = Matmul20001269Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001264InputAtom::run(stensor20001264_async_copy_buf, dtensor10000280_tile_ptr, thread_idx);
    STensor20001263InputAtom::run(stensor20001263_async_copy_buf, dtensor10000279_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001264InputAtom::run(stensor20001264_ptr, dtensor10000280_tile_ptr + 4096*(for_idx+1), thread_idx);
        STensor20001263InputAtom::run(stensor20001263_ptr, dtensor10000279_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001264_ptr, stensor20001264_async_copy_buf);
      SWAP(stensor20001263_ptr, stensor20001263_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001266Kernel::get_mma_rC(thread_idx);
      Matmul20001266Kernel::run(mma_rC, stensor20001262_ptr, stensor20001263_ptr, (char*)(buf+0), thread_idx);
      Matmul20001266Kernel::write_back_mma_rC(stensor20001266_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001267_ptr, stensor20001266_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001269Kernel::run(matmul_20001269_accum, stensor20001266_ptr, stensor20001264_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001269Kernel::write_back_mma_rC(stensor20001269_ptr, matmul_20001269_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001267OutputAtom::run(dtensor10000281_tile_ptr, stensor20001267_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20001269OutputAtom::run(dtensor10000282_tile_ptr, stensor20001269_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000283_ptr, half_t const* __restrict__ dtensor10000281_ptr, half_t const* __restrict__ dtensor10000282_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001283_ptr = (half_t*)(buf + 128);
  half_t *stensor20001282_ptr = (half_t*)(buf + 16544);
  half_t *stensor20001280_ptr = (half_t*)(buf + 16512);
  half_t *stensor30001277_ptr = (half_t*)(buf + 41088);
  half_t *stensor20001277_ptr = (half_t*)(buf + 32896);
  half_t *stensor20001278_ptr = (half_t*)(buf + 24704);
  half_t *stensor30001278_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001281_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001279_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000281 -> stensor 20001277
  const half_t *dtensor10000281_tile_ptr = dtensor10000281_ptr  + blockIdx.x*1*262144 + blockIdx.y*16*1;
  using DTensor10000281TileLayout = Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001277InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>, DTensor10000281TileLayout, NUM_THREADS>;
  half_t *stensor20001277_async_copy_buf = stensor30001277_ptr;
  // Copy for G->S: dtensor 10000282 -> stensor 20001278
  const half_t *dtensor10000282_tile_ptr = dtensor10000282_ptr  + blockIdx.x*1*262144 + blockIdx.y*16*1;
  using DTensor10000282TileLayout = Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001278InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>, DTensor10000282TileLayout, NUM_THREADS>;
  half_t *stensor20001278_async_copy_buf = stensor30001278_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001283 -> dtensor 10000283
  half_t *dtensor10000283_tile_ptr = dtensor10000283_ptr  + blockIdx.x*1*16384 + blockIdx.y*16*64;
  using DTensor10000283TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001283OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000283TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001279_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001281_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20001278InputAtom::run(stensor20001278_async_copy_buf, dtensor10000282_tile_ptr, thread_idx);
    STensor20001277InputAtom::run(stensor20001277_async_copy_buf, dtensor10000281_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001278InputAtom::run(stensor20001278_ptr, dtensor10000282_tile_ptr + 65536*(for_idx+1), thread_idx);
        STensor20001277InputAtom::run(stensor20001277_ptr, dtensor10000281_tile_ptr + 65536*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001278_ptr, stensor20001278_async_copy_buf);
      SWAP(stensor20001277_ptr, stensor20001277_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>, Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>, NUM_THREADS>;
      Kernel::run(stensor20001279_ptr, stensor20001277_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>, Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>, NUM_THREADS>;
      Kernel::run(stensor20001281_ptr, stensor20001278_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001280_ptr, stensor20001279_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001282_ptr, stensor20001281_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using In1Layout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<1024>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001283_ptr, stensor20001282_ptr, stensor20001280_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001283OutputAtom::run(dtensor10000283_tile_ptr, stensor20001283_ptr, thread_idx);
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
    half_t *dtensor10000281 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000282 = (half_t*)((char*)buf + 1048576);
    half_t *dtensor10000278 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000279 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000280 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 57472;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 57472);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000281, dtensor10000282, dtensor10000278, dtensor10000279, dtensor10000280);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000283 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000281 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000282 = (half_t*)((char*)buf + 1048576);
    dim3 grid_dim(2, 16, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 49280;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 49280);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000283, dtensor10000281, dtensor10000282);
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
