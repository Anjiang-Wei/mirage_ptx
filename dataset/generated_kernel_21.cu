#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000305_ptr, half_t* __restrict__ dtensor10000306_ptr, half_t const* __restrict__ dtensor10000302_ptr, half_t const* __restrict__ dtensor10000303_ptr, half_t const* __restrict__ dtensor10000304_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001389_ptr = (half_t*)(buf + 128);
  half_t *stensor20001386_ptr = (half_t*)(buf + 49280);
  half_t *stensor30001384_ptr = (half_t*)(buf + 41088);
  half_t *stensor20001384_ptr = (half_t*)(buf + 32896);
  half_t *stensor30001383_ptr = (half_t*)(buf + 24704);
  half_t *stensor20001383_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001387_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001382_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000302 -> stensor 20001382
  const half_t *dtensor10000302_tile_ptr = dtensor10000302_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000302TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001382InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000302TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000303 -> stensor 20001383
  const half_t *dtensor10000303_tile_ptr = dtensor10000303_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*1;
  using DTensor10000303TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001383InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000303TileLayout, NUM_THREADS>;
  half_t *stensor20001383_async_copy_buf = stensor30001383_ptr;
  // Copy for G->S: dtensor 10000304 -> stensor 20001384
  const half_t *dtensor10000304_tile_ptr = dtensor10000304_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*64;
  using DTensor10000304TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001384InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000304TileLayout, NUM_THREADS>;
  half_t *stensor20001384_async_copy_buf = stensor30001384_ptr;
  
  STensor20001382InputAtom::run(stensor20001382_ptr, dtensor10000302_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001387 -> dtensor 10000305
  half_t *dtensor10000305_tile_ptr = dtensor10000305_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*256 + blockIdx.z*64*1;
  using DTensor10000305TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001387OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000305TileLayout, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20001389 -> dtensor 10000306
  half_t *dtensor10000306_tile_ptr = dtensor10000306_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*256 + blockIdx.z*64*1;
  using DTensor10000306TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001389OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000306TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001387_ptr, thread_idx);
  
  
  using Matmul20001386LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001386LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001386LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001386LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001386LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001386Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001386LayoutA, Matmul20001386LayoutB, Matmul20001386LayoutC, Matmul20001386LayoutAAligned, Matmul20001386LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001389LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001389LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001389LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001389LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001389LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001389Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001389LayoutA, Matmul20001389LayoutB, Matmul20001389LayoutC, Matmul20001389LayoutAAligned, Matmul20001389LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001389_accum = Matmul20001389Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001384InputAtom::run(stensor20001384_async_copy_buf, dtensor10000304_tile_ptr, thread_idx);
    STensor20001383InputAtom::run(stensor20001383_async_copy_buf, dtensor10000303_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001384InputAtom::run(stensor20001384_ptr, dtensor10000304_tile_ptr + 4096*(for_idx+1), thread_idx);
        STensor20001383InputAtom::run(stensor20001383_ptr, dtensor10000303_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001384_ptr, stensor20001384_async_copy_buf);
      SWAP(stensor20001383_ptr, stensor20001383_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001386Kernel::get_mma_rC(thread_idx);
      Matmul20001386Kernel::run(mma_rC, stensor20001382_ptr, stensor20001383_ptr, (char*)(buf+0), thread_idx);
      Matmul20001386Kernel::write_back_mma_rC(stensor20001386_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001387_ptr, stensor20001386_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001389Kernel::run(matmul_20001389_accum, stensor20001386_ptr, stensor20001384_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001389Kernel::write_back_mma_rC(stensor20001389_ptr, matmul_20001389_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001387OutputAtom::run(dtensor10000305_tile_ptr, stensor20001387_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20001389OutputAtom::run(dtensor10000306_tile_ptr, stensor20001389_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000307_ptr, half_t const* __restrict__ dtensor10000305_ptr, half_t const* __restrict__ dtensor10000306_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001403_ptr = (half_t*)(buf + 128);
  half_t *stensor20001402_ptr = (half_t*)(buf + 8336);
  half_t *stensor20001400_ptr = (half_t*)(buf + 8320);
  half_t *stensor30001397_ptr = (half_t*)(buf + 20608);
  half_t *stensor20001397_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001398_ptr = (half_t*)(buf + 12416);
  half_t *stensor30001398_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001401_ptr = (half_t*)(buf + 4224);
  half_t *stensor20001399_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000305 -> stensor 20001397
  const half_t *dtensor10000305_tile_ptr = dtensor10000305_ptr  + blockIdx.x*1*262144 + blockIdx.y*8*1;
  using DTensor10000305TileLayout = Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001397InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>, DTensor10000305TileLayout, NUM_THREADS>;
  half_t *stensor20001397_async_copy_buf = stensor30001397_ptr;
  // Copy for G->S: dtensor 10000306 -> stensor 20001398
  const half_t *dtensor10000306_tile_ptr = dtensor10000306_ptr  + blockIdx.x*1*262144 + blockIdx.y*8*1;
  using DTensor10000306TileLayout = Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001398InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>, DTensor10000306TileLayout, NUM_THREADS>;
  half_t *stensor20001398_async_copy_buf = stensor30001398_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001403 -> dtensor 10000307
  half_t *dtensor10000307_tile_ptr = dtensor10000307_ptr  + blockIdx.x*1*16384 + blockIdx.y*8*64;
  using DTensor10000307TileLayout = Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001403OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000307TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<512>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001399_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001401_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20001398InputAtom::run(stensor20001398_async_copy_buf, dtensor10000306_tile_ptr, thread_idx);
    STensor20001397InputAtom::run(stensor20001397_async_copy_buf, dtensor10000305_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001398InputAtom::run(stensor20001398_ptr, dtensor10000306_tile_ptr + 65536*(for_idx+1), thread_idx);
        STensor20001397InputAtom::run(stensor20001397_ptr, dtensor10000305_tile_ptr + 65536*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001398_ptr, stensor20001398_async_copy_buf);
      SWAP(stensor20001397_ptr, stensor20001397_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>, Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>, NUM_THREADS>;
      Kernel::run(stensor20001399_ptr, stensor20001397_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>, Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>, NUM_THREADS>;
      Kernel::run(stensor20001401_ptr, stensor20001398_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001400_ptr, stensor20001399_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001402_ptr, stensor20001401_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using In1Layout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<512>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001403_ptr, stensor20001402_ptr, stensor20001400_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001403OutputAtom::run(dtensor10000307_tile_ptr, stensor20001403_ptr, thread_idx);
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
    half_t *dtensor10000305 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000306 = (half_t*)((char*)buf + 1048576);
    half_t *dtensor10000302 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000303 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000304 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 57472;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 57472);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000305, dtensor10000306, dtensor10000302, dtensor10000303, dtensor10000304);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000307 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000305 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000306 = (half_t*)((char*)buf + 1048576);
    dim3 grid_dim(2, 32, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 24704;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 24704);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000307, dtensor10000305, dtensor10000306);
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
