#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000257_ptr, half_t* __restrict__ dtensor10000258_ptr, half_t const* __restrict__ dtensor10000254_ptr, half_t const* __restrict__ dtensor10000255_ptr, half_t const* __restrict__ dtensor10000256_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001145_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001143_ptr = (half_t*)(buf + 128);
  half_t *stensor20001141_ptr = (half_t*)(buf + 71808);
  half_t *stensor30001138_ptr = (half_t*)(buf + 55424);
  half_t *stensor20001138_ptr = (half_t*)(buf + 39040);
  half_t *stensor20001139_ptr = (half_t*)(buf + 22656);
  half_t *stensor30001139_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001142_ptr = (half_t*)(buf + 2176);
  half_t *stensor20001137_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000254 -> stensor 20001137
  const half_t *dtensor10000254_tile_ptr = dtensor10000254_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000254TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001137InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000254TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000255 -> stensor 20001138
  const half_t *dtensor10000255_tile_ptr = dtensor10000255_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000255TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001138InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000255TileLayout, NUM_THREADS>;
  half_t *stensor20001138_async_copy_buf = stensor30001138_ptr;
  // Copy for G->S: dtensor 10000256 -> stensor 20001139
  const half_t *dtensor10000256_tile_ptr = dtensor10000256_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000256TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001139InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000256TileLayout, NUM_THREADS>;
  half_t *stensor20001139_async_copy_buf = stensor30001139_ptr;
  
  STensor20001137InputAtom::run(stensor20001137_ptr, dtensor10000254_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001145 -> dtensor 10000258
  half_t *dtensor10000258_tile_ptr = dtensor10000258_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*1 + blockIdx.z*16*512;
  using DTensor10000258TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20001145OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000258TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20001143 -> dtensor 10000257
  half_t *dtensor10000257_tile_ptr = dtensor10000257_ptr  + blockIdx.x*1*1 + blockIdx.y*64*8 + blockIdx.z*16*4096;
  using DTensor10000257TileLayout = Layout<Shape<Int<1>, Int<64>, Int<16>>, Stride<Int<1>, Int<8>, Int<4096>>>;
  using STensor20001143OutputAtom = tb::OutputNonChunkedSyncCopy<half_t, DTensor10000257TileLayout, Layout<Shape<Int<1>, Int<64>, Int<16>>, Stride<Int<1>, Int<1>, Int<64>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001142_ptr, thread_idx);
  
  
  using Matmul20001141LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001141LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001141LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001141LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001141LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001141Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001141LayoutA, Matmul20001141LayoutB, Matmul20001141LayoutC, Matmul20001141LayoutAAligned, Matmul20001141LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001145LayoutA = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001145LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001145LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001145LayoutAAligned = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001145LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001145Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001145LayoutA, Matmul20001145LayoutB, Matmul20001145LayoutC, Matmul20001145LayoutAAligned, Matmul20001145LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001145_accum = Matmul20001145Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001139InputAtom::run(stensor20001139_async_copy_buf, dtensor10000256_tile_ptr, thread_idx);
    STensor20001138InputAtom::run(stensor20001138_async_copy_buf, dtensor10000255_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001139InputAtom::run(stensor20001139_ptr, dtensor10000256_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20001138InputAtom::run(stensor20001138_ptr, dtensor10000255_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001139_ptr, stensor20001139_async_copy_buf);
      SWAP(stensor20001138_ptr, stensor20001138_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001141Kernel::get_mma_rC(thread_idx);
      Matmul20001141Kernel::run(mma_rC, stensor20001137_ptr, stensor20001138_ptr, (char*)(buf+0), thread_idx);
      Matmul20001141Kernel::write_back_mma_rC(stensor20001141_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<128>, Int<16>, Int<1>>, Stride<Int<1>, Int<128>, Int<1>>>, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>, Int<1>>, Stride<Int<16>, Int<1>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001142_ptr, stensor20001141_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001145Kernel::run(matmul_20001145_accum, stensor20001141_ptr, stensor20001139_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001145Kernel::write_back_mma_rC(stensor20001145_ptr, matmul_20001145_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<1>, Int<128>, Int<16>>, Stride<Int<1>, Int<1>, Int<128>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<16>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001143_ptr, stensor20001142_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20001145OutputAtom::run(dtensor10000258_tile_ptr, stensor20001145_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001143OutputAtom::run(dtensor10000257_tile_ptr, stensor20001143_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000259_ptr, half_t const* __restrict__ dtensor10000257_ptr, half_t const* __restrict__ dtensor10000258_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001159_ptr = (half_t*)(buf + 128);
  half_t *stensor20001158_ptr = (half_t*)(buf + 2192);
  half_t *stensor20001156_ptr = (half_t*)(buf + 2176);
  half_t *stensor20001153_ptr = (half_t*)(buf + 4224);
  half_t *stensor30001154_ptr = (half_t*)(buf + 3200);
  half_t *stensor20001154_ptr = (half_t*)(buf + 2176);
  half_t *stensor20001157_ptr = (half_t*)(buf + 1152);
  half_t *stensor20001155_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000257 -> stensor 20001153
  const half_t *dtensor10000257_tile_ptr = dtensor10000257_ptr  + blockIdx.x*1*1 + blockIdx.y*4*4096;
  using DTensor10000257TileLayout = Layout<Shape<Int<1>, Int<128>, Int<4>>, Stride<Int<1>, Int<8>, Int<4096>>>;
  using STensor20001153InputAtom = tb::InputNonChunkedSyncCopy<half_t, Layout<Shape<Int<1>, Int<128>, Int<4>>, Stride<Int<1>, Int<1>, Int<128>>>, DTensor10000257TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000258 -> stensor 20001154
  const half_t *dtensor10000258_tile_ptr = dtensor10000258_ptr  + blockIdx.x*1*131072 + blockIdx.y*4*512;
  using DTensor10000258TileLayout = Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20001154InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<128>, Int<512>>>, DTensor10000258TileLayout, NUM_THREADS>;
  half_t *stensor20001154_async_copy_buf = stensor30001154_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001159 -> dtensor 10000259
  half_t *dtensor10000259_tile_ptr = dtensor10000259_ptr  + blockIdx.x*1*16384 + blockIdx.y*4*64;
  using DTensor10000259TileLayout = Layout<Shape<Int<64>, Int<4>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001159OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000259TileLayout, Layout<Shape<Int<64>, Int<4>, Int<1>>, Stride<Int<1>, Int<64>, Int<256>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 512, NUM_THREADS>::run(stensor20001155_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 512, NUM_THREADS>::run(stensor20001157_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20001154InputAtom::run(stensor20001154_async_copy_buf, dtensor10000258_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001154InputAtom::run(stensor20001154_ptr, dtensor10000258_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001154_ptr, stensor20001154_async_copy_buf);
    }
    {
      // OP type: tb_input_op
      STensor20001153InputAtom::run(stensor20001153_ptr, dtensor10000257_tile_ptr + 1024*for_idx, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<1>, Int<128>, Int<4>>, Stride<Int<1>, Int<1>, Int<128>>>, Layout<Shape<Int<1>, Int<128>, Int<4>>, Stride<Int<1>, Int<1>, Int<128>>>, NUM_THREADS>;
      Kernel::run(stensor20001155_ptr, stensor20001153_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<128>, Int<1>>>, Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<128>, Int<512>>>, NUM_THREADS>;
      Kernel::run(stensor20001157_ptr, stensor20001154_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<1>, Int<128>, Int<4>>, Stride<Int<1>, Int<1>, Int<128>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<1>, Int<4>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001156_ptr, stensor20001155_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<1>, Int<128>, Int<4>>, Stride<Int<1>, Int<1>, Int<128>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<4>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001158_ptr, stensor20001157_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<1>, Int<64>, Int<4>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<4>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<4>>, Stride<Int<256>, Int<1>, Int<64>>>;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001159_ptr, stensor20001158_ptr, stensor20001156_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001159OutputAtom::run(dtensor10000259_tile_ptr, stensor20001159_ptr, thread_idx);
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
    half_t *dtensor10000257 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000258 = (half_t*)((char*)buf + 2097152);
    half_t *dtensor10000254 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000255 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000256 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000257, dtensor10000258, dtensor10000254, dtensor10000255, dtensor10000256);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000259 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000257 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000258 = (half_t*)((char*)buf + 2097152);
    dim3 grid_dim(2, 64, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 5248;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 5248);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000259, dtensor10000257, dtensor10000258);
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
