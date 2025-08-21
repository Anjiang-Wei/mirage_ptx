#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000275_ptr, half_t* __restrict__ dtensor10000276_ptr, half_t const* __restrict__ dtensor10000272_ptr, half_t const* __restrict__ dtensor10000273_ptr, half_t const* __restrict__ dtensor10000274_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001239_ptr = (half_t*)(buf + 128);
  half_t *stensor20001236_ptr = (half_t*)(buf + 49280);
  half_t *stensor30001234_ptr = (half_t*)(buf + 41088);
  half_t *stensor20001234_ptr = (half_t*)(buf + 32896);
  half_t *stensor30001233_ptr = (half_t*)(buf + 24704);
  half_t *stensor20001233_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001237_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001232_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000272 -> stensor 20001232
  const half_t *dtensor10000272_tile_ptr = dtensor10000272_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000272TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001232InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000272TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000273 -> stensor 20001233
  const half_t *dtensor10000273_tile_ptr = dtensor10000273_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*1;
  using DTensor10000273TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001233InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000273TileLayout, NUM_THREADS>;
  half_t *stensor20001233_async_copy_buf = stensor30001233_ptr;
  // Copy for G->S: dtensor 10000274 -> stensor 20001234
  const half_t *dtensor10000274_tile_ptr = dtensor10000274_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*64;
  using DTensor10000274TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001234InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000274TileLayout, NUM_THREADS>;
  half_t *stensor20001234_async_copy_buf = stensor30001234_ptr;
  
  STensor20001232InputAtom::run(stensor20001232_ptr, dtensor10000272_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001237 -> dtensor 10000275
  half_t *dtensor10000275_tile_ptr = dtensor10000275_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*256 + blockIdx.z*64*1;
  using DTensor10000275TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001237OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000275TileLayout, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20001239 -> dtensor 10000276
  half_t *dtensor10000276_tile_ptr = dtensor10000276_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*256 + blockIdx.z*64*1;
  using DTensor10000276TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001239OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000276TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001237_ptr, thread_idx);
  
  
  using Matmul20001236LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001236LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001236LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001236LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001236LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001236Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001236LayoutA, Matmul20001236LayoutB, Matmul20001236LayoutC, Matmul20001236LayoutAAligned, Matmul20001236LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001239LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001239LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001239LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001239LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001239LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001239Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001239LayoutA, Matmul20001239LayoutB, Matmul20001239LayoutC, Matmul20001239LayoutAAligned, Matmul20001239LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001239_accum = Matmul20001239Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001234InputAtom::run(stensor20001234_async_copy_buf, dtensor10000274_tile_ptr, thread_idx);
    STensor20001233InputAtom::run(stensor20001233_async_copy_buf, dtensor10000273_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001234InputAtom::run(stensor20001234_ptr, dtensor10000274_tile_ptr + 4096*(for_idx+1), thread_idx);
        STensor20001233InputAtom::run(stensor20001233_ptr, dtensor10000273_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001234_ptr, stensor20001234_async_copy_buf);
      SWAP(stensor20001233_ptr, stensor20001233_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001236Kernel::get_mma_rC(thread_idx);
      Matmul20001236Kernel::run(mma_rC, stensor20001232_ptr, stensor20001233_ptr, (char*)(buf+0), thread_idx);
      Matmul20001236Kernel::write_back_mma_rC(stensor20001236_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001237_ptr, stensor20001236_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001239Kernel::run(matmul_20001239_accum, stensor20001236_ptr, stensor20001234_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001239Kernel::write_back_mma_rC(stensor20001239_ptr, matmul_20001239_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001237OutputAtom::run(dtensor10000275_tile_ptr, stensor20001237_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20001239OutputAtom::run(dtensor10000276_tile_ptr, stensor20001239_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000277_ptr, half_t const* __restrict__ dtensor10000275_ptr, half_t const* __restrict__ dtensor10000276_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001253_ptr = (half_t*)(buf + 2208);
  half_t *stensor20001252_ptr = (half_t*)(buf + 160);
  half_t *stensor20001250_ptr = (half_t*)(buf + 128);
  half_t *stensor20001251_ptr = (half_t*)(buf + 98432);
  half_t *stensor20001249_ptr = (half_t*)(buf + 65664);
  half_t *stensor20001248_ptr = (half_t*)(buf + 32896);
  half_t *stensor20001247_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000275 -> stensor 20001247
  const half_t *dtensor10000275_tile_ptr = dtensor10000275_ptr  + blockIdx.x*1*262144 + blockIdx.y*16*1;
  using DTensor10000275TileLayout = Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001247InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, DTensor10000275TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000276 -> stensor 20001248
  const half_t *dtensor10000276_tile_ptr = dtensor10000276_ptr  + blockIdx.x*1*262144 + blockIdx.y*16*1;
  using DTensor10000276TileLayout = Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001248InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, DTensor10000276TileLayout, NUM_THREADS>;
  
  STensor20001247InputAtom::run(stensor20001247_ptr, dtensor10000275_tile_ptr, thread_idx);
  STensor20001248InputAtom::run(stensor20001248_ptr, dtensor10000276_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001253 -> dtensor 10000277
  half_t *dtensor10000277_tile_ptr = dtensor10000277_ptr  + blockIdx.x*1*16384 + blockIdx.y*16*64;
  using DTensor10000277TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001253OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000277TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 16384, NUM_THREADS>::run(stensor20001249_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 16384, NUM_THREADS>::run(stensor20001251_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, NUM_THREADS>;
      Kernel::run(stensor20001249_ptr, stensor20001247_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, NUM_THREADS>;
      Kernel::run(stensor20001251_ptr, stensor20001248_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001250_ptr, stensor20001249_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001252_ptr, stensor20001251_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using In1Layout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<1024>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001253_ptr, stensor20001252_ptr, stensor20001250_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001253OutputAtom::run(dtensor10000277_tile_ptr, stensor20001253_ptr, thread_idx);
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
    half_t *dtensor10000275 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000276 = (half_t*)((char*)buf + 1048576);
    half_t *dtensor10000272 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000273 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000274 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 57472;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 57472);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000275, dtensor10000276, dtensor10000272, dtensor10000273, dtensor10000274);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000277 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000275 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000276 = (half_t*)((char*)buf + 1048576);
    dim3 grid_dim(2, 16, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 131200;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 131200);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000277, dtensor10000275, dtensor10000276);
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
