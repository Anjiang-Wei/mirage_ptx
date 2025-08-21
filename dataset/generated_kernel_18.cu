#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000287_ptr, half_t* __restrict__ dtensor10000288_ptr, half_t const* __restrict__ dtensor10000284_ptr, half_t const* __restrict__ dtensor10000285_ptr, half_t const* __restrict__ dtensor10000286_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001299_ptr = (half_t*)(buf + 128);
  half_t *stensor20001296_ptr = (half_t*)(buf + 49280);
  half_t *stensor30001294_ptr = (half_t*)(buf + 41088);
  half_t *stensor20001294_ptr = (half_t*)(buf + 32896);
  half_t *stensor30001293_ptr = (half_t*)(buf + 24704);
  half_t *stensor20001293_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001297_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001292_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000284 -> stensor 20001292
  const half_t *dtensor10000284_tile_ptr = dtensor10000284_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000284TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001292InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000284TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000285 -> stensor 20001293
  const half_t *dtensor10000285_tile_ptr = dtensor10000285_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*1;
  using DTensor10000285TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001293InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000285TileLayout, NUM_THREADS>;
  half_t *stensor20001293_async_copy_buf = stensor30001293_ptr;
  // Copy for G->S: dtensor 10000286 -> stensor 20001294
  const half_t *dtensor10000286_tile_ptr = dtensor10000286_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*64;
  using DTensor10000286TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001294InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000286TileLayout, NUM_THREADS>;
  half_t *stensor20001294_async_copy_buf = stensor30001294_ptr;
  
  STensor20001292InputAtom::run(stensor20001292_ptr, dtensor10000284_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001297 -> dtensor 10000287
  half_t *dtensor10000287_tile_ptr = dtensor10000287_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*1 + blockIdx.z*64*1024;
  using DTensor10000287TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20001297OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000287TileLayout, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20001299 -> dtensor 10000288
  half_t *dtensor10000288_tile_ptr = dtensor10000288_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*1 + blockIdx.z*64*1024;
  using DTensor10000288TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20001299OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000288TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001297_ptr, thread_idx);
  
  
  using Matmul20001296LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001296LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001296LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001296LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001296LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001296Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001296LayoutA, Matmul20001296LayoutB, Matmul20001296LayoutC, Matmul20001296LayoutAAligned, Matmul20001296LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001299LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001299LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001299LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001299LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001299LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001299Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001299LayoutA, Matmul20001299LayoutB, Matmul20001299LayoutC, Matmul20001299LayoutAAligned, Matmul20001299LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001299_accum = Matmul20001299Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001294InputAtom::run(stensor20001294_async_copy_buf, dtensor10000286_tile_ptr, thread_idx);
    STensor20001293InputAtom::run(stensor20001293_async_copy_buf, dtensor10000285_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001294InputAtom::run(stensor20001294_ptr, dtensor10000286_tile_ptr + 4096*(for_idx+1), thread_idx);
        STensor20001293InputAtom::run(stensor20001293_ptr, dtensor10000285_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001294_ptr, stensor20001294_async_copy_buf);
      SWAP(stensor20001293_ptr, stensor20001293_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001296Kernel::get_mma_rC(thread_idx);
      Matmul20001296Kernel::run(mma_rC, stensor20001292_ptr, stensor20001293_ptr, (char*)(buf+0), thread_idx);
      Matmul20001296Kernel::write_back_mma_rC(stensor20001296_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001297_ptr, stensor20001296_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001299Kernel::run(matmul_20001299_accum, stensor20001296_ptr, stensor20001294_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001299Kernel::write_back_mma_rC(stensor20001299_ptr, matmul_20001299_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001297OutputAtom::run(dtensor10000287_tile_ptr, stensor20001297_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20001299OutputAtom::run(dtensor10000288_tile_ptr, stensor20001299_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000289_ptr, half_t const* __restrict__ dtensor10000287_ptr, half_t const* __restrict__ dtensor10000288_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001313_ptr = (half_t*)(buf + 400);
  half_t *stensor20001312_ptr = (half_t*)(buf + 144);
  half_t *stensor20001310_ptr = (half_t*)(buf + 128);
  half_t *stensor20001311_ptr = (half_t*)(buf + 12416);
  half_t *stensor20001309_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001308_ptr = (half_t*)(buf + 4224);
  half_t *stensor20001307_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000287 -> stensor 20001307
  const half_t *dtensor10000287_tile_ptr = dtensor10000287_ptr  + blockIdx.x*1*262144 + blockIdx.y*2*1024;
  using DTensor10000287TileLayout = Layout<Shape<Int<1024>, Int<2>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20001307InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<1024>, Int<2>, Int<1>>, Stride<Int<1>, Int<1024>, Int<2048>>>, DTensor10000287TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000288 -> stensor 20001308
  const half_t *dtensor10000288_tile_ptr = dtensor10000288_ptr  + blockIdx.x*1*262144 + blockIdx.y*2*1024;
  using DTensor10000288TileLayout = Layout<Shape<Int<1024>, Int<2>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20001308InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<1024>, Int<2>, Int<1>>, Stride<Int<1>, Int<1024>, Int<2048>>>, DTensor10000288TileLayout, NUM_THREADS>;
  
  STensor20001307InputAtom::run(stensor20001307_ptr, dtensor10000287_tile_ptr, thread_idx);
  STensor20001308InputAtom::run(stensor20001308_ptr, dtensor10000288_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001313 -> dtensor 10000289
  half_t *dtensor10000289_tile_ptr = dtensor10000289_ptr  + blockIdx.x*1*16384 + blockIdx.y*2*64;
  using DTensor10000289TileLayout = Layout<Shape<Int<64>, Int<2>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001313OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000289TileLayout, Layout<Shape<Int<64>, Int<2>, Int<1>>, Stride<Int<1>, Int<64>, Int<128>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001309_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001311_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<1>, Int<1024>, Int<2>>, Stride<Int<1>, Int<1>, Int<1024>>>, Layout<Shape<Int<1>, Int<1024>, Int<2>>, Stride<Int<2048>, Int<1>, Int<1024>>>, NUM_THREADS>;
      Kernel::run(stensor20001309_ptr, stensor20001307_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<1>, Int<1024>, Int<2>>, Stride<Int<1>, Int<1>, Int<1024>>>, Layout<Shape<Int<1>, Int<1024>, Int<2>>, Stride<Int<2048>, Int<1>, Int<1024>>>, NUM_THREADS>;
      Kernel::run(stensor20001311_ptr, stensor20001308_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<1>, Int<1024>, Int<2>>, Stride<Int<1>, Int<1>, Int<1024>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<1>, Int<2>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001310_ptr, stensor20001309_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<1>, Int<1024>, Int<2>>, Stride<Int<1>, Int<1>, Int<1024>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001312_ptr, stensor20001311_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<2>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<128>, Int<1>, Int<64>>>;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001313_ptr, stensor20001312_ptr, stensor20001310_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001313OutputAtom::run(dtensor10000289_tile_ptr, stensor20001313_ptr, thread_idx);
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
    half_t *dtensor10000287 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000288 = (half_t*)((char*)buf + 1048576);
    half_t *dtensor10000284 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000285 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000286 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 57472;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 57472);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000287, dtensor10000288, dtensor10000284, dtensor10000285, dtensor10000286);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000289 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000287 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000288 = (half_t*)((char*)buf + 1048576);
    dim3 grid_dim(2, 128, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 16512;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 16512);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000289, dtensor10000287, dtensor10000288);
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
