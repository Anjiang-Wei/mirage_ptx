#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000245_ptr, half_t* __restrict__ dtensor10000246_ptr, half_t const* __restrict__ dtensor10000242_ptr, half_t const* __restrict__ dtensor10000243_ptr, half_t const* __restrict__ dtensor10000244_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001081_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001079_ptr = (half_t*)(buf + 128);
  half_t *stensor20001077_ptr = (half_t*)(buf + 71808);
  half_t *stensor30001074_ptr = (half_t*)(buf + 55424);
  half_t *stensor20001074_ptr = (half_t*)(buf + 39040);
  half_t *stensor20001075_ptr = (half_t*)(buf + 22656);
  half_t *stensor30001075_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001078_ptr = (half_t*)(buf + 2176);
  half_t *stensor20001073_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000242 -> stensor 20001073
  const half_t *dtensor10000242_tile_ptr = dtensor10000242_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000242TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001073InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000242TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000243 -> stensor 20001074
  const half_t *dtensor10000243_tile_ptr = dtensor10000243_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000243TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001074InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000243TileLayout, NUM_THREADS>;
  half_t *stensor20001074_async_copy_buf = stensor30001074_ptr;
  // Copy for G->S: dtensor 10000244 -> stensor 20001075
  const half_t *dtensor10000244_tile_ptr = dtensor10000244_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000244TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001075InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000244TileLayout, NUM_THREADS>;
  half_t *stensor20001075_async_copy_buf = stensor30001075_ptr;
  
  STensor20001073InputAtom::run(stensor20001073_ptr, dtensor10000242_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001081 -> dtensor 10000246
  half_t *dtensor10000246_tile_ptr = dtensor10000246_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*256 + blockIdx.z*16*1;
  using DTensor10000246TileLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20001081OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000246TileLayout, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20001079 -> dtensor 10000245
  half_t *dtensor10000245_tile_ptr = dtensor10000245_ptr  + blockIdx.x*1*2048 + blockIdx.y*1*256 + blockIdx.z*16*1;
  using DTensor10000245TileLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<256>, Int<2048>>>;
  using STensor20001079OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000245TileLayout, Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001078_ptr, thread_idx);
  
  
  using Matmul20001077LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001077LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001077LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001077LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001077LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001077Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001077LayoutA, Matmul20001077LayoutB, Matmul20001077LayoutC, Matmul20001077LayoutAAligned, Matmul20001077LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001081LayoutA = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001081LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001081LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001081LayoutAAligned = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001081LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001081Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001081LayoutA, Matmul20001081LayoutB, Matmul20001081LayoutC, Matmul20001081LayoutAAligned, Matmul20001081LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001081_accum = Matmul20001081Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001075InputAtom::run(stensor20001075_async_copy_buf, dtensor10000244_tile_ptr, thread_idx);
    STensor20001074InputAtom::run(stensor20001074_async_copy_buf, dtensor10000243_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001075InputAtom::run(stensor20001075_ptr, dtensor10000244_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20001074InputAtom::run(stensor20001074_ptr, dtensor10000243_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001075_ptr, stensor20001075_async_copy_buf);
      SWAP(stensor20001074_ptr, stensor20001074_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001077Kernel::get_mma_rC(thread_idx);
      Matmul20001077Kernel::run(mma_rC, stensor20001073_ptr, stensor20001074_ptr, (char*)(buf+0), thread_idx);
      Matmul20001077Kernel::write_back_mma_rC(stensor20001077_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001078_ptr, stensor20001077_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001081Kernel::run(matmul_20001081_accum, stensor20001077_ptr, stensor20001075_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001081Kernel::write_back_mma_rC(stensor20001081_ptr, matmul_20001081_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001079_ptr, stensor20001078_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20001081OutputAtom::run(dtensor10000246_tile_ptr, stensor20001081_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001079OutputAtom::run(dtensor10000245_tile_ptr, stensor20001079_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000247_ptr, half_t const* __restrict__ dtensor10000245_ptr, half_t const* __restrict__ dtensor10000246_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001095_ptr = (half_t*)(buf + 128);
  half_t *stensor20001094_ptr = (half_t*)(buf + 2224);
  half_t *stensor20001092_ptr = (half_t*)(buf + 2208);
  half_t *stensor30001089_ptr = (half_t*)(buf + 6336);
  half_t *stensor20001089_ptr = (half_t*)(buf + 6304);
  half_t *stensor20001090_ptr = (half_t*)(buf + 4256);
  half_t *stensor30001090_ptr = (half_t*)(buf + 2208);
  half_t *stensor20001093_ptr = (half_t*)(buf + 160);
  half_t *stensor20001091_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000245 -> stensor 20001089
  const half_t *dtensor10000245_tile_ptr = dtensor10000245_ptr  + blockIdx.x*1*2048 + blockIdx.y*8*1;
  using DTensor10000245TileLayout = Layout<Shape<Int<8>, Int<2>, Int<1>>, Stride<Int<1>, Int<256>, Int<2048>>>;
  using STensor20001089InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<8>, Int<2>, Int<1>>, Stride<Int<1>, Int<8>, Int<16>>>, DTensor10000245TileLayout, NUM_THREADS>;
  half_t *stensor20001089_async_copy_buf = stensor30001089_ptr;
  // Copy for G->S: dtensor 10000246 -> stensor 20001090
  const half_t *dtensor10000246_tile_ptr = dtensor10000246_ptr  + blockIdx.x*1*131072 + blockIdx.y*8*1;
  using DTensor10000246TileLayout = Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20001090InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>, DTensor10000246TileLayout, NUM_THREADS>;
  half_t *stensor20001090_async_copy_buf = stensor30001090_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001095 -> dtensor 10000247
  half_t *dtensor10000247_tile_ptr = dtensor10000247_ptr  + blockIdx.x*1*16384 + blockIdx.y*8*64;
  using DTensor10000247TileLayout = Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001095OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000247TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<512>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 16, NUM_THREADS>::run(stensor20001091_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 1024, NUM_THREADS>::run(stensor20001093_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20001090InputAtom::run(stensor20001090_async_copy_buf, dtensor10000246_tile_ptr, thread_idx);
    STensor20001089InputAtom::run(stensor20001089_async_copy_buf, dtensor10000245_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001090InputAtom::run(stensor20001090_ptr, dtensor10000246_tile_ptr + 32768*(for_idx+1), thread_idx);
        STensor20001089InputAtom::run(stensor20001089_ptr, dtensor10000245_tile_ptr + 512*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001090_ptr, stensor20001090_async_copy_buf);
      SWAP(stensor20001089_ptr, stensor20001089_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<2>, Int<1>>, Stride<Int<1>, Int<8>, Int<16>>>, Layout<Shape<Int<8>, Int<2>, Int<1>>, Stride<Int<1>, Int<8>, Int<16>>>, NUM_THREADS>;
      Kernel::run(stensor20001091_ptr, stensor20001089_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>, Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>, NUM_THREADS>;
      Kernel::run(stensor20001093_ptr, stensor20001090_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<8>, Int<2>, Int<1>>, Stride<Int<1>, Int<8>, Int<16>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001092_ptr, stensor20001091_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001094_ptr, stensor20001093_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using In1Layout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<512>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001095_ptr, stensor20001094_ptr, stensor20001092_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001095OutputAtom::run(dtensor10000247_tile_ptr, stensor20001095_ptr, thread_idx);
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
    half_t *dtensor10000245 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000246 = (half_t*)((char*)buf + 8192);
    half_t *dtensor10000242 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000243 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000244 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000245, dtensor10000246, dtensor10000242, dtensor10000243, dtensor10000244);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000247 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000245 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000246 = (half_t*)((char*)buf + 8192);
    dim3 grid_dim(2, 32, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 6368;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 6368);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000247, dtensor10000245, dtensor10000246);
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
