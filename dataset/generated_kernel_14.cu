#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000263_ptr, half_t* __restrict__ dtensor10000264_ptr, half_t const* __restrict__ dtensor10000260_ptr, half_t const* __restrict__ dtensor10000261_ptr, half_t const* __restrict__ dtensor10000262_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001177_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001175_ptr = (half_t*)(buf + 128);
  half_t *stensor20001173_ptr = (half_t*)(buf + 71808);
  half_t *stensor30001170_ptr = (half_t*)(buf + 55424);
  half_t *stensor20001170_ptr = (half_t*)(buf + 39040);
  half_t *stensor20001171_ptr = (half_t*)(buf + 22656);
  half_t *stensor30001171_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001174_ptr = (half_t*)(buf + 2176);
  half_t *stensor20001169_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000260 -> stensor 20001169
  const half_t *dtensor10000260_tile_ptr = dtensor10000260_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000260TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001169InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000260TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000261 -> stensor 20001170
  const half_t *dtensor10000261_tile_ptr = dtensor10000261_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000261TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001170InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000261TileLayout, NUM_THREADS>;
  half_t *stensor20001170_async_copy_buf = stensor30001170_ptr;
  // Copy for G->S: dtensor 10000262 -> stensor 20001171
  const half_t *dtensor10000262_tile_ptr = dtensor10000262_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000262TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001171InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000262TileLayout, NUM_THREADS>;
  half_t *stensor20001171_async_copy_buf = stensor30001171_ptr;
  
  STensor20001169InputAtom::run(stensor20001169_ptr, dtensor10000260_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001177 -> dtensor 10000264
  half_t *dtensor10000264_tile_ptr = dtensor10000264_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*256 + blockIdx.z*16*1;
  using DTensor10000264TileLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20001177OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000264TileLayout, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20001175 -> dtensor 10000263
  half_t *dtensor10000263_tile_ptr = dtensor10000263_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*256 + blockIdx.z*16*1;
  using DTensor10000263TileLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20001175OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000263TileLayout, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001174_ptr, thread_idx);
  
  
  using Matmul20001173LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001173LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001173LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001173LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001173LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001173Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001173LayoutA, Matmul20001173LayoutB, Matmul20001173LayoutC, Matmul20001173LayoutAAligned, Matmul20001173LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001177LayoutA = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001177LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001177LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001177LayoutAAligned = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001177LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001177Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001177LayoutA, Matmul20001177LayoutB, Matmul20001177LayoutC, Matmul20001177LayoutAAligned, Matmul20001177LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001177_accum = Matmul20001177Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001171InputAtom::run(stensor20001171_async_copy_buf, dtensor10000262_tile_ptr, thread_idx);
    STensor20001170InputAtom::run(stensor20001170_async_copy_buf, dtensor10000261_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001171InputAtom::run(stensor20001171_ptr, dtensor10000262_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20001170InputAtom::run(stensor20001170_ptr, dtensor10000261_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001171_ptr, stensor20001171_async_copy_buf);
      SWAP(stensor20001170_ptr, stensor20001170_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001173Kernel::get_mma_rC(thread_idx);
      Matmul20001173Kernel::run(mma_rC, stensor20001169_ptr, stensor20001170_ptr, (char*)(buf+0), thread_idx);
      Matmul20001173Kernel::write_back_mma_rC(stensor20001173_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001174_ptr, stensor20001173_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001177Kernel::run(matmul_20001177_accum, stensor20001173_ptr, stensor20001171_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001177Kernel::write_back_mma_rC(stensor20001177_ptr, matmul_20001177_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001175_ptr, stensor20001174_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20001177OutputAtom::run(dtensor10000264_tile_ptr, stensor20001177_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001175OutputAtom::run(dtensor10000263_tile_ptr, stensor20001175_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000265_ptr, half_t const* __restrict__ dtensor10000263_ptr, half_t const* __restrict__ dtensor10000264_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001191_ptr = (half_t*)(buf + 1168);
  half_t *stensor20001190_ptr = (half_t*)(buf + 144);
  half_t *stensor20001188_ptr = (half_t*)(buf + 128);
  half_t *stensor20001189_ptr = (half_t*)(buf + 24704);
  half_t *stensor20001187_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001186_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001185_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000263 -> stensor 20001185
  const half_t *dtensor10000263_tile_ptr = dtensor10000263_ptr  + blockIdx.x*1*131072 + blockIdx.y*8*1;
  using DTensor10000263TileLayout = Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20001185InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>, DTensor10000263TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000264 -> stensor 20001186
  const half_t *dtensor10000264_tile_ptr = dtensor10000264_ptr  + blockIdx.x*1*131072 + blockIdx.y*8*1;
  using DTensor10000264TileLayout = Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20001186InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>, DTensor10000264TileLayout, NUM_THREADS>;
  
  STensor20001185InputAtom::run(stensor20001185_ptr, dtensor10000263_tile_ptr, thread_idx);
  STensor20001186InputAtom::run(stensor20001186_ptr, dtensor10000264_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001191 -> dtensor 10000265
  half_t *dtensor10000265_tile_ptr = dtensor10000265_ptr  + blockIdx.x*1*16384 + blockIdx.y*8*64;
  using DTensor10000265TileLayout = Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001191OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000265TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<512>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001187_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001189_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>, Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>, NUM_THREADS>;
      Kernel::run(stensor20001187_ptr, stensor20001185_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>, Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>, NUM_THREADS>;
      Kernel::run(stensor20001189_ptr, stensor20001186_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001188_ptr, stensor20001187_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001190_ptr, stensor20001189_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using In1Layout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<512>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001191_ptr, stensor20001190_ptr, stensor20001188_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001191OutputAtom::run(dtensor10000265_tile_ptr, stensor20001191_ptr, thread_idx);
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
    half_t *dtensor10000263 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000264 = (half_t*)((char*)buf + 524288);
    half_t *dtensor10000260 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000261 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000262 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000263, dtensor10000264, dtensor10000260, dtensor10000261, dtensor10000262);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000265 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000263 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000264 = (half_t*)((char*)buf + 524288);
    dim3 grid_dim(2, 32, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 32896;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 32896);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000265, dtensor10000263, dtensor10000264);
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
