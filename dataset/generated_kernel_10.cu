#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000239_ptr, half_t* __restrict__ dtensor10000240_ptr, half_t const* __restrict__ dtensor10000236_ptr, half_t const* __restrict__ dtensor10000237_ptr, half_t const* __restrict__ dtensor10000238_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001049_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001047_ptr = (half_t*)(buf + 128);
  half_t *stensor20001045_ptr = (half_t*)(buf + 71808);
  half_t *stensor30001042_ptr = (half_t*)(buf + 55424);
  half_t *stensor20001042_ptr = (half_t*)(buf + 39040);
  half_t *stensor20001043_ptr = (half_t*)(buf + 22656);
  half_t *stensor30001043_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001046_ptr = (half_t*)(buf + 2176);
  half_t *stensor20001041_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000236 -> stensor 20001041
  const half_t *dtensor10000236_tile_ptr = dtensor10000236_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000236TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001041InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000236TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000237 -> stensor 20001042
  const half_t *dtensor10000237_tile_ptr = dtensor10000237_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000237TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001042InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000237TileLayout, NUM_THREADS>;
  half_t *stensor20001042_async_copy_buf = stensor30001042_ptr;
  // Copy for G->S: dtensor 10000238 -> stensor 20001043
  const half_t *dtensor10000238_tile_ptr = dtensor10000238_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000238TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001043InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000238TileLayout, NUM_THREADS>;
  half_t *stensor20001043_async_copy_buf = stensor30001043_ptr;
  
  STensor20001041InputAtom::run(stensor20001041_ptr, dtensor10000236_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001049 -> dtensor 10000240
  half_t *dtensor10000240_tile_ptr = dtensor10000240_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*256 + blockIdx.z*16*1;
  using DTensor10000240TileLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20001049OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000240TileLayout, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20001047 -> dtensor 10000239
  half_t *dtensor10000239_tile_ptr = dtensor10000239_ptr  + blockIdx.x*1*2048 + blockIdx.y*1*256 + blockIdx.z*16*1;
  using DTensor10000239TileLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<256>, Int<2048>>>;
  using STensor20001047OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000239TileLayout, Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001046_ptr, thread_idx);
  
  
  using Matmul20001045LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001045LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001045LayoutC = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001045LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001045LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001045Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001045LayoutA, Matmul20001045LayoutB, Matmul20001045LayoutC, Matmul20001045LayoutAAligned, Matmul20001045LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001049LayoutA = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001049LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001049LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001049LayoutAAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001049LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001049Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001049LayoutA, Matmul20001049LayoutB, Matmul20001049LayoutC, Matmul20001049LayoutAAligned, Matmul20001049LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001049_accum = Matmul20001049Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001043InputAtom::run(stensor20001043_async_copy_buf, dtensor10000238_tile_ptr, thread_idx);
    STensor20001042InputAtom::run(stensor20001042_async_copy_buf, dtensor10000237_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001043InputAtom::run(stensor20001043_ptr, dtensor10000238_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20001042InputAtom::run(stensor20001042_ptr, dtensor10000237_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001043_ptr, stensor20001043_async_copy_buf);
      SWAP(stensor20001042_ptr, stensor20001042_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001045Kernel::get_mma_rC(thread_idx);
      Matmul20001045Kernel::run(mma_rC, stensor20001041_ptr, stensor20001042_ptr, (char*)(buf+0), thread_idx);
      Matmul20001045Kernel::write_back_mma_rC(stensor20001045_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<128>, Int<1>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001046_ptr, stensor20001045_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001049Kernel::run(matmul_20001049_accum, stensor20001045_ptr, stensor20001043_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001049Kernel::write_back_mma_rC(stensor20001049_ptr, matmul_20001049_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001047_ptr, stensor20001046_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20001049OutputAtom::run(dtensor10000240_tile_ptr, stensor20001049_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001047OutputAtom::run(dtensor10000239_tile_ptr, stensor20001047_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000241_ptr, half_t const* __restrict__ dtensor10000239_ptr, half_t const* __restrict__ dtensor10000240_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001063_ptr = (half_t*)(buf + 1168);
  half_t *stensor20001062_ptr = (half_t*)(buf + 144);
  half_t *stensor20001060_ptr = (half_t*)(buf + 128);
  half_t *stensor20001061_ptr = (half_t*)(buf + 8576);
  half_t *stensor20001059_ptr = (half_t*)(buf + 8448);
  half_t *stensor20001058_ptr = (half_t*)(buf + 256);
  half_t *stensor20001057_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000239 -> stensor 20001057
  const half_t *dtensor10000239_tile_ptr = dtensor10000239_ptr  + blockIdx.x*1*2048 + blockIdx.y*8*1;
  using DTensor10000239TileLayout = Layout<Shape<Int<8>, Int<8>, Int<1>>, Stride<Int<1>, Int<256>, Int<2048>>>;
  using STensor20001057InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<8>, Int<8>, Int<1>>, Stride<Int<1>, Int<8>, Int<64>>>, DTensor10000239TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000240 -> stensor 20001058
  const half_t *dtensor10000240_tile_ptr = dtensor10000240_ptr  + blockIdx.x*1*131072 + blockIdx.y*8*1;
  using DTensor10000240TileLayout = Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20001058InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>, DTensor10000240TileLayout, NUM_THREADS>;
  
  STensor20001057InputAtom::run(stensor20001057_ptr, dtensor10000239_tile_ptr, thread_idx);
  STensor20001058InputAtom::run(stensor20001058_ptr, dtensor10000240_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001063 -> dtensor 10000241
  half_t *dtensor10000241_tile_ptr = dtensor10000241_ptr  + blockIdx.x*1*16384 + blockIdx.y*8*64;
  using DTensor10000241TileLayout = Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001063OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000241TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<512>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 64, NUM_THREADS>::run(stensor20001059_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001061_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<8>, Int<1>>, Stride<Int<1>, Int<8>, Int<64>>>, Layout<Shape<Int<8>, Int<8>, Int<1>>, Stride<Int<1>, Int<8>, Int<64>>>, NUM_THREADS>;
      Kernel::run(stensor20001059_ptr, stensor20001057_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>, Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>, NUM_THREADS>;
      Kernel::run(stensor20001061_ptr, stensor20001058_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<8>, Int<8>, Int<1>>, Stride<Int<1>, Int<8>, Int<64>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001060_ptr, stensor20001059_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001062_ptr, stensor20001061_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using In1Layout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<512>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001063_ptr, stensor20001062_ptr, stensor20001060_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001063OutputAtom::run(dtensor10000241_tile_ptr, stensor20001063_ptr, thread_idx);
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
    half_t *dtensor10000239 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000240 = (half_t*)((char*)buf + 8192);
    half_t *dtensor10000236 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000237 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000238 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000239, dtensor10000240, dtensor10000236, dtensor10000237, dtensor10000238);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000241 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000239 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000240 = (half_t*)((char*)buf + 8192);
    dim3 grid_dim(2, 32, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 16768;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 16768);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000241, dtensor10000239, dtensor10000240);
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
