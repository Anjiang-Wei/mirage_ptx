#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000317_ptr, half_t* __restrict__ dtensor10000318_ptr, half_t const* __restrict__ dtensor10000314_ptr, half_t const* __restrict__ dtensor10000315_ptr, half_t const* __restrict__ dtensor10000316_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001453_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001451_ptr = (half_t*)(buf + 128);
  half_t *stensor20001449_ptr = (half_t*)(buf + 49280);
  half_t *stensor30001446_ptr = (half_t*)(buf + 41088);
  half_t *stensor20001446_ptr = (half_t*)(buf + 32896);
  half_t *stensor20001447_ptr = (half_t*)(buf + 24704);
  half_t *stensor30001447_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001450_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001445_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000314 -> stensor 20001445
  const half_t *dtensor10000314_tile_ptr = dtensor10000314_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000314TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001445InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000314TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000315 -> stensor 20001446
  const half_t *dtensor10000315_tile_ptr = dtensor10000315_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*1;
  using DTensor10000315TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001446InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000315TileLayout, NUM_THREADS>;
  half_t *stensor20001446_async_copy_buf = stensor30001446_ptr;
  // Copy for G->S: dtensor 10000316 -> stensor 20001447
  const half_t *dtensor10000316_tile_ptr = dtensor10000316_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*64;
  using DTensor10000316TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001447InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000316TileLayout, NUM_THREADS>;
  half_t *stensor20001447_async_copy_buf = stensor30001447_ptr;
  
  STensor20001445InputAtom::run(stensor20001445_ptr, dtensor10000314_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001453 -> dtensor 10000318
  half_t *dtensor10000318_tile_ptr = dtensor10000318_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*256 + blockIdx.z*64*1;
  using DTensor10000318TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001453OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000318TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20001451 -> dtensor 10000317
  half_t *dtensor10000317_tile_ptr = dtensor10000317_ptr  + blockIdx.x*1*4096 + blockIdx.y*1*256 + blockIdx.z*64*1;
  using DTensor10000317TileLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20001451OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000317TileLayout, Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001450_ptr, thread_idx);
  
  
  using Matmul20001449LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001449LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001449LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001449LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001449LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001449Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001449LayoutA, Matmul20001449LayoutB, Matmul20001449LayoutC, Matmul20001449LayoutAAligned, Matmul20001449LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001453LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001453LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001453LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001453LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001453LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001453Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001453LayoutA, Matmul20001453LayoutB, Matmul20001453LayoutC, Matmul20001453LayoutAAligned, Matmul20001453LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001453_accum = Matmul20001453Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001447InputAtom::run(stensor20001447_async_copy_buf, dtensor10000316_tile_ptr, thread_idx);
    STensor20001446InputAtom::run(stensor20001446_async_copy_buf, dtensor10000315_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001447InputAtom::run(stensor20001447_ptr, dtensor10000316_tile_ptr + 4096*(for_idx+1), thread_idx);
        STensor20001446InputAtom::run(stensor20001446_ptr, dtensor10000315_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001447_ptr, stensor20001447_async_copy_buf);
      SWAP(stensor20001446_ptr, stensor20001446_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001449Kernel::get_mma_rC(thread_idx);
      Matmul20001449Kernel::run(mma_rC, stensor20001445_ptr, stensor20001446_ptr, (char*)(buf+0), thread_idx);
      Matmul20001449Kernel::write_back_mma_rC(stensor20001449_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001450_ptr, stensor20001449_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001453Kernel::run(matmul_20001453_accum, stensor20001449_ptr, stensor20001447_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001453Kernel::write_back_mma_rC(stensor20001453_ptr, matmul_20001453_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001451_ptr, stensor20001450_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20001453OutputAtom::run(dtensor10000318_tile_ptr, stensor20001453_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001451OutputAtom::run(dtensor10000317_tile_ptr, stensor20001451_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000319_ptr, half_t const* __restrict__ dtensor10000317_ptr, half_t const* __restrict__ dtensor10000318_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001467_ptr = (half_t*)(buf + 128);
  half_t *stensor20001466_ptr = (half_t*)(buf + 8480);
  half_t *stensor20001464_ptr = (half_t*)(buf + 8448);
  half_t *stensor30001461_ptr = (half_t*)(buf + 24960);
  half_t *stensor20001461_ptr = (half_t*)(buf + 24832);
  half_t *stensor20001462_ptr = (half_t*)(buf + 16640);
  half_t *stensor30001462_ptr = (half_t*)(buf + 8448);
  half_t *stensor20001465_ptr = (half_t*)(buf + 256);
  half_t *stensor20001463_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000317 -> stensor 20001461
  const half_t *dtensor10000317_tile_ptr = dtensor10000317_ptr  + blockIdx.x*1*4096 + blockIdx.y*16*1;
  using DTensor10000317TileLayout = Layout<Shape<Int<16>, Int<4>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20001461InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<16>, Int<4>, Int<1>>, Stride<Int<1>, Int<16>, Int<64>>>, DTensor10000317TileLayout, NUM_THREADS>;
  half_t *stensor20001461_async_copy_buf = stensor30001461_ptr;
  // Copy for G->S: dtensor 10000318 -> stensor 20001462
  const half_t *dtensor10000318_tile_ptr = dtensor10000318_ptr  + blockIdx.x*1*262144 + blockIdx.y*16*1;
  using DTensor10000318TileLayout = Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001462InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>, DTensor10000318TileLayout, NUM_THREADS>;
  half_t *stensor20001462_async_copy_buf = stensor30001462_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001467 -> dtensor 10000319
  half_t *dtensor10000319_tile_ptr = dtensor10000319_ptr  + blockIdx.x*1*16384 + blockIdx.y*16*64;
  using DTensor10000319TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001467OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000319TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 64, NUM_THREADS>::run(stensor20001463_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001465_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20001462InputAtom::run(stensor20001462_async_copy_buf, dtensor10000318_tile_ptr, thread_idx);
    STensor20001461InputAtom::run(stensor20001461_async_copy_buf, dtensor10000317_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001462InputAtom::run(stensor20001462_ptr, dtensor10000318_tile_ptr + 65536*(for_idx+1), thread_idx);
        STensor20001461InputAtom::run(stensor20001461_ptr, dtensor10000317_tile_ptr + 1024*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001462_ptr, stensor20001462_async_copy_buf);
      SWAP(stensor20001461_ptr, stensor20001461_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<4>, Int<1>>, Stride<Int<1>, Int<16>, Int<64>>>, Layout<Shape<Int<16>, Int<4>, Int<1>>, Stride<Int<1>, Int<16>, Int<64>>>, NUM_THREADS>;
      Kernel::run(stensor20001463_ptr, stensor20001461_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>, Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>, NUM_THREADS>;
      Kernel::run(stensor20001465_ptr, stensor20001462_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<16>, Int<4>, Int<1>>, Stride<Int<1>, Int<16>, Int<64>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001464_ptr, stensor20001463_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001466_ptr, stensor20001465_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using In1Layout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<1024>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001467_ptr, stensor20001466_ptr, stensor20001464_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001467OutputAtom::run(dtensor10000319_tile_ptr, stensor20001467_ptr, thread_idx);
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
    half_t *dtensor10000317 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000318 = (half_t*)((char*)buf + 16384);
    half_t *dtensor10000314 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000315 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000316 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 57472;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 57472);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000317, dtensor10000318, dtensor10000314, dtensor10000315, dtensor10000316);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000319 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000317 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000318 = (half_t*)((char*)buf + 16384);
    dim3 grid_dim(2, 16, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 25088;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 25088);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000319, dtensor10000317, dtensor10000318);
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
