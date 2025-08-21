#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000335_ptr, half_t* __restrict__ dtensor10000336_ptr, half_t const* __restrict__ dtensor10000332_ptr, half_t const* __restrict__ dtensor10000333_ptr, half_t const* __restrict__ dtensor10000334_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001549_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001547_ptr = (half_t*)(buf + 128);
  half_t *stensor20001545_ptr = (half_t*)(buf + 49280);
  half_t *stensor30001542_ptr = (half_t*)(buf + 41088);
  half_t *stensor20001542_ptr = (half_t*)(buf + 32896);
  half_t *stensor20001543_ptr = (half_t*)(buf + 24704);
  half_t *stensor30001543_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001546_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001541_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000332 -> stensor 20001541
  const half_t *dtensor10000332_tile_ptr = dtensor10000332_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000332TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001541InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000332TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000333 -> stensor 20001542
  const half_t *dtensor10000333_tile_ptr = dtensor10000333_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*1;
  using DTensor10000333TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001542InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000333TileLayout, NUM_THREADS>;
  half_t *stensor20001542_async_copy_buf = stensor30001542_ptr;
  // Copy for G->S: dtensor 10000334 -> stensor 20001543
  const half_t *dtensor10000334_tile_ptr = dtensor10000334_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*64;
  using DTensor10000334TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001543InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000334TileLayout, NUM_THREADS>;
  half_t *stensor20001543_async_copy_buf = stensor30001543_ptr;
  
  STensor20001541InputAtom::run(stensor20001541_ptr, dtensor10000332_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001549 -> dtensor 10000336
  half_t *dtensor10000336_tile_ptr = dtensor10000336_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*256 + blockIdx.z*64*1;
  using DTensor10000336TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001549OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000336TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20001547 -> dtensor 10000335
  half_t *dtensor10000335_tile_ptr = dtensor10000335_ptr  + blockIdx.x*1*4096 + blockIdx.y*1*256 + blockIdx.z*64*1;
  using DTensor10000335TileLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20001547OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000335TileLayout, Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001546_ptr, thread_idx);
  
  
  using Matmul20001545LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001545LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001545LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001545LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001545LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001545Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001545LayoutA, Matmul20001545LayoutB, Matmul20001545LayoutC, Matmul20001545LayoutAAligned, Matmul20001545LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001549LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001549LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001549LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001549LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001549LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001549Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001549LayoutA, Matmul20001549LayoutB, Matmul20001549LayoutC, Matmul20001549LayoutAAligned, Matmul20001549LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001549_accum = Matmul20001549Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001543InputAtom::run(stensor20001543_async_copy_buf, dtensor10000334_tile_ptr, thread_idx);
    STensor20001542InputAtom::run(stensor20001542_async_copy_buf, dtensor10000333_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001543InputAtom::run(stensor20001543_ptr, dtensor10000334_tile_ptr + 4096*(for_idx+1), thread_idx);
        STensor20001542InputAtom::run(stensor20001542_ptr, dtensor10000333_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001543_ptr, stensor20001543_async_copy_buf);
      SWAP(stensor20001542_ptr, stensor20001542_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001545Kernel::get_mma_rC(thread_idx);
      Matmul20001545Kernel::run(mma_rC, stensor20001541_ptr, stensor20001542_ptr, (char*)(buf+0), thread_idx);
      Matmul20001545Kernel::write_back_mma_rC(stensor20001545_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001546_ptr, stensor20001545_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001549Kernel::run(matmul_20001549_accum, stensor20001545_ptr, stensor20001543_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001549Kernel::write_back_mma_rC(stensor20001549_ptr, matmul_20001549_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001547_ptr, stensor20001546_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20001549OutputAtom::run(dtensor10000336_tile_ptr, stensor20001549_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001547OutputAtom::run(dtensor10000335_tile_ptr, stensor20001547_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000337_ptr, half_t const* __restrict__ dtensor10000335_ptr, half_t const* __restrict__ dtensor10000336_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001563_ptr = (half_t*)(buf + 1168);
  half_t *stensor20001562_ptr = (half_t*)(buf + 144);
  half_t *stensor20001560_ptr = (half_t*)(buf + 128);
  half_t *stensor20001561_ptr = (half_t*)(buf + 17024);
  half_t *stensor20001559_ptr = (half_t*)(buf + 16768);
  half_t *stensor20001558_ptr = (half_t*)(buf + 384);
  half_t *stensor20001557_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000335 -> stensor 20001557
  const half_t *dtensor10000335_tile_ptr = dtensor10000335_ptr  + blockIdx.x*1*4096 + blockIdx.y*8*1;
  using DTensor10000335TileLayout = Layout<Shape<Int<8>, Int<16>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20001557InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<8>, Int<16>, Int<1>>, Stride<Int<1>, Int<8>, Int<128>>>, DTensor10000335TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000336 -> stensor 20001558
  const half_t *dtensor10000336_tile_ptr = dtensor10000336_ptr  + blockIdx.x*1*262144 + blockIdx.y*8*1;
  using DTensor10000336TileLayout = Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001558InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>, DTensor10000336TileLayout, NUM_THREADS>;
  
  STensor20001557InputAtom::run(stensor20001557_ptr, dtensor10000335_tile_ptr, thread_idx);
  STensor20001558InputAtom::run(stensor20001558_ptr, dtensor10000336_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001563 -> dtensor 10000337
  half_t *dtensor10000337_tile_ptr = dtensor10000337_ptr  + blockIdx.x*1*16384 + blockIdx.y*8*64;
  using DTensor10000337TileLayout = Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001563OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000337TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<512>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 128, NUM_THREADS>::run(stensor20001559_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 8192, NUM_THREADS>::run(stensor20001561_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<16>, Int<1>>, Stride<Int<1>, Int<8>, Int<128>>>, Layout<Shape<Int<8>, Int<16>, Int<1>>, Stride<Int<1>, Int<8>, Int<128>>>, NUM_THREADS>;
      Kernel::run(stensor20001559_ptr, stensor20001557_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>, Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>, NUM_THREADS>;
      Kernel::run(stensor20001561_ptr, stensor20001558_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<8>, Int<16>, Int<1>>, Stride<Int<1>, Int<8>, Int<128>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001560_ptr, stensor20001559_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001562_ptr, stensor20001561_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using In1Layout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<512>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001563_ptr, stensor20001562_ptr, stensor20001560_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001563OutputAtom::run(dtensor10000337_tile_ptr, stensor20001563_ptr, thread_idx);
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
    half_t *dtensor10000335 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000336 = (half_t*)((char*)buf + 16384);
    half_t *dtensor10000332 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000333 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000334 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 57472;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 57472);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000335, dtensor10000336, dtensor10000332, dtensor10000333, dtensor10000334);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000337 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000335 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000336 = (half_t*)((char*)buf + 16384);
    dim3 grid_dim(2, 32, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 33408;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 33408);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000337, dtensor10000335, dtensor10000336);
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
