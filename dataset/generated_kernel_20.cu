#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000299_ptr, half_t* __restrict__ dtensor10000300_ptr, half_t const* __restrict__ dtensor10000296_ptr, half_t const* __restrict__ dtensor10000297_ptr, half_t const* __restrict__ dtensor10000298_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001359_ptr = (half_t*)(buf + 128);
  half_t *stensor20001356_ptr = (half_t*)(buf + 49280);
  half_t *stensor30001354_ptr = (half_t*)(buf + 41088);
  half_t *stensor20001354_ptr = (half_t*)(buf + 32896);
  half_t *stensor30001353_ptr = (half_t*)(buf + 24704);
  half_t *stensor20001353_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001357_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001352_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000296 -> stensor 20001352
  const half_t *dtensor10000296_tile_ptr = dtensor10000296_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000296TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001352InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000296TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000297 -> stensor 20001353
  const half_t *dtensor10000297_tile_ptr = dtensor10000297_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*1;
  using DTensor10000297TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001353InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000297TileLayout, NUM_THREADS>;
  half_t *stensor20001353_async_copy_buf = stensor30001353_ptr;
  // Copy for G->S: dtensor 10000298 -> stensor 20001354
  const half_t *dtensor10000298_tile_ptr = dtensor10000298_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*64;
  using DTensor10000298TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001354InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000298TileLayout, NUM_THREADS>;
  half_t *stensor20001354_async_copy_buf = stensor30001354_ptr;
  
  STensor20001352InputAtom::run(stensor20001352_ptr, dtensor10000296_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001357 -> dtensor 10000299
  half_t *dtensor10000299_tile_ptr = dtensor10000299_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*256 + blockIdx.z*64*1;
  using DTensor10000299TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001357OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000299TileLayout, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20001359 -> dtensor 10000300
  half_t *dtensor10000300_tile_ptr = dtensor10000300_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*256 + blockIdx.z*64*1;
  using DTensor10000300TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001359OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000300TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001357_ptr, thread_idx);
  
  
  using Matmul20001356LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001356LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001356LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001356LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001356LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001356Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001356LayoutA, Matmul20001356LayoutB, Matmul20001356LayoutC, Matmul20001356LayoutAAligned, Matmul20001356LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001359LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001359LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001359LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001359LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001359LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001359Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001359LayoutA, Matmul20001359LayoutB, Matmul20001359LayoutC, Matmul20001359LayoutAAligned, Matmul20001359LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001359_accum = Matmul20001359Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001354InputAtom::run(stensor20001354_async_copy_buf, dtensor10000298_tile_ptr, thread_idx);
    STensor20001353InputAtom::run(stensor20001353_async_copy_buf, dtensor10000297_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001354InputAtom::run(stensor20001354_ptr, dtensor10000298_tile_ptr + 4096*(for_idx+1), thread_idx);
        STensor20001353InputAtom::run(stensor20001353_ptr, dtensor10000297_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001354_ptr, stensor20001354_async_copy_buf);
      SWAP(stensor20001353_ptr, stensor20001353_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001356Kernel::get_mma_rC(thread_idx);
      Matmul20001356Kernel::run(mma_rC, stensor20001352_ptr, stensor20001353_ptr, (char*)(buf+0), thread_idx);
      Matmul20001356Kernel::write_back_mma_rC(stensor20001356_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001357_ptr, stensor20001356_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001359Kernel::run(matmul_20001359_accum, stensor20001356_ptr, stensor20001354_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001359Kernel::write_back_mma_rC(stensor20001359_ptr, matmul_20001359_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001357OutputAtom::run(dtensor10000299_tile_ptr, stensor20001357_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20001359OutputAtom::run(dtensor10000300_tile_ptr, stensor20001359_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000301_ptr, half_t const* __restrict__ dtensor10000299_ptr, half_t const* __restrict__ dtensor10000300_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001373_ptr = (half_t*)(buf + 1168);
  half_t *stensor20001372_ptr = (half_t*)(buf + 144);
  half_t *stensor20001370_ptr = (half_t*)(buf + 128);
  half_t *stensor20001371_ptr = (half_t*)(buf + 49280);
  half_t *stensor20001369_ptr = (half_t*)(buf + 32896);
  half_t *stensor20001368_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001367_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000299 -> stensor 20001367
  const half_t *dtensor10000299_tile_ptr = dtensor10000299_ptr  + blockIdx.x*1*262144 + blockIdx.y*8*1;
  using DTensor10000299TileLayout = Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001367InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>, DTensor10000299TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000300 -> stensor 20001368
  const half_t *dtensor10000300_tile_ptr = dtensor10000300_ptr  + blockIdx.x*1*262144 + blockIdx.y*8*1;
  using DTensor10000300TileLayout = Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001368InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>, DTensor10000300TileLayout, NUM_THREADS>;
  
  STensor20001367InputAtom::run(stensor20001367_ptr, dtensor10000299_tile_ptr, thread_idx);
  STensor20001368InputAtom::run(stensor20001368_ptr, dtensor10000300_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001373 -> dtensor 10000301
  half_t *dtensor10000301_tile_ptr = dtensor10000301_ptr  + blockIdx.x*1*16384 + blockIdx.y*8*64;
  using DTensor10000301TileLayout = Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001373OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000301TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<512>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 8192, NUM_THREADS>::run(stensor20001369_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 8192, NUM_THREADS>::run(stensor20001371_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>, Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>, NUM_THREADS>;
      Kernel::run(stensor20001369_ptr, stensor20001367_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>, Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>, NUM_THREADS>;
      Kernel::run(stensor20001371_ptr, stensor20001368_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001370_ptr, stensor20001369_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001372_ptr, stensor20001371_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using In1Layout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<512>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001373_ptr, stensor20001372_ptr, stensor20001370_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001373OutputAtom::run(dtensor10000301_tile_ptr, stensor20001373_ptr, thread_idx);
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
    half_t *dtensor10000299 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000300 = (half_t*)((char*)buf + 1048576);
    half_t *dtensor10000296 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000297 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000298 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 57472;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 57472);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000299, dtensor10000300, dtensor10000296, dtensor10000297, dtensor10000298);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000301 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000299 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000300 = (half_t*)((char*)buf + 1048576);
    dim3 grid_dim(2, 32, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 65664;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 65664);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000301, dtensor10000299, dtensor10000300);
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
