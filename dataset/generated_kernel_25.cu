#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000329_ptr, half_t* __restrict__ dtensor10000330_ptr, half_t const* __restrict__ dtensor10000326_ptr, half_t const* __restrict__ dtensor10000327_ptr, half_t const* __restrict__ dtensor10000328_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001517_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001515_ptr = (half_t*)(buf + 128);
  half_t *stensor20001513_ptr = (half_t*)(buf + 49280);
  half_t *stensor30001510_ptr = (half_t*)(buf + 41088);
  half_t *stensor20001510_ptr = (half_t*)(buf + 32896);
  half_t *stensor20001511_ptr = (half_t*)(buf + 24704);
  half_t *stensor30001511_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001514_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001509_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000326 -> stensor 20001509
  const half_t *dtensor10000326_tile_ptr = dtensor10000326_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000326TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001509InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000326TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000327 -> stensor 20001510
  const half_t *dtensor10000327_tile_ptr = dtensor10000327_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*1;
  using DTensor10000327TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001510InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000327TileLayout, NUM_THREADS>;
  half_t *stensor20001510_async_copy_buf = stensor30001510_ptr;
  // Copy for G->S: dtensor 10000328 -> stensor 20001511
  const half_t *dtensor10000328_tile_ptr = dtensor10000328_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*64;
  using DTensor10000328TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001511InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000328TileLayout, NUM_THREADS>;
  half_t *stensor20001511_async_copy_buf = stensor30001511_ptr;
  
  STensor20001509InputAtom::run(stensor20001509_ptr, dtensor10000326_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001517 -> dtensor 10000330
  half_t *dtensor10000330_tile_ptr = dtensor10000330_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*1 + blockIdx.z*64*1024;
  using DTensor10000330TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20001517OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000330TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20001515 -> dtensor 10000329
  half_t *dtensor10000329_tile_ptr = dtensor10000329_ptr  + blockIdx.x*1*4096 + blockIdx.y*1*256 + blockIdx.z*64*1;
  using DTensor10000329TileLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20001515OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000329TileLayout, Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001514_ptr, thread_idx);
  
  
  using Matmul20001513LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001513LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001513LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001513LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001513LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001513Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001513LayoutA, Matmul20001513LayoutB, Matmul20001513LayoutC, Matmul20001513LayoutAAligned, Matmul20001513LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001517LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001517LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001517LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001517LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001517LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001517Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001517LayoutA, Matmul20001517LayoutB, Matmul20001517LayoutC, Matmul20001517LayoutAAligned, Matmul20001517LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001517_accum = Matmul20001517Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001511InputAtom::run(stensor20001511_async_copy_buf, dtensor10000328_tile_ptr, thread_idx);
    STensor20001510InputAtom::run(stensor20001510_async_copy_buf, dtensor10000327_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001511InputAtom::run(stensor20001511_ptr, dtensor10000328_tile_ptr + 4096*(for_idx+1), thread_idx);
        STensor20001510InputAtom::run(stensor20001510_ptr, dtensor10000327_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001511_ptr, stensor20001511_async_copy_buf);
      SWAP(stensor20001510_ptr, stensor20001510_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001513Kernel::get_mma_rC(thread_idx);
      Matmul20001513Kernel::run(mma_rC, stensor20001509_ptr, stensor20001510_ptr, (char*)(buf+0), thread_idx);
      Matmul20001513Kernel::write_back_mma_rC(stensor20001513_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001514_ptr, stensor20001513_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001517Kernel::run(matmul_20001517_accum, stensor20001513_ptr, stensor20001511_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001517Kernel::write_back_mma_rC(stensor20001517_ptr, matmul_20001517_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001515_ptr, stensor20001514_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20001517OutputAtom::run(dtensor10000330_tile_ptr, stensor20001517_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001515OutputAtom::run(dtensor10000329_tile_ptr, stensor20001515_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000331_ptr, half_t const* __restrict__ dtensor10000329_ptr, half_t const* __restrict__ dtensor10000330_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001531_ptr = (half_t*)(buf + 128);
  half_t *stensor20001530_ptr = (half_t*)(buf + 1232);
  half_t *stensor20001528_ptr = (half_t*)(buf + 1216);
  half_t *stensor20001525_ptr = (half_t*)(buf + 3264);
  half_t *stensor30001526_ptr = (half_t*)(buf + 2240);
  half_t *stensor20001526_ptr = (half_t*)(buf + 1216);
  half_t *stensor20001529_ptr = (half_t*)(buf + 192);
  half_t *stensor20001527_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000329 -> stensor 20001525
  const half_t *dtensor10000329_tile_ptr = dtensor10000329_ptr  + blockIdx.x*1*4096 + blockIdx.y*2*1;
  using DTensor10000329TileLayout = Layout<Shape<Int<2>, Int<4>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20001525InputAtom = tb::InputNonChunkedSyncCopy<half_t, Layout<Shape<Int<2>, Int<4>, Int<1>>, Stride<Int<1>, Int<8>, Int<32>>>, DTensor10000329TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000330 -> stensor 20001526
  const half_t *dtensor10000330_tile_ptr = dtensor10000330_ptr  + blockIdx.x*1*262144 + blockIdx.y*2*1024;
  using DTensor10000330TileLayout = Layout<Shape<Int<256>, Int<2>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20001526InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<256>, Int<2>, Int<1>>, Stride<Int<1>, Int<256>, Int<512>>>, DTensor10000330TileLayout, NUM_THREADS>;
  half_t *stensor20001526_async_copy_buf = stensor30001526_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001531 -> dtensor 10000331
  half_t *dtensor10000331_tile_ptr = dtensor10000331_ptr  + blockIdx.x*1*16384 + blockIdx.y*2*64;
  using DTensor10000331TileLayout = Layout<Shape<Int<64>, Int<2>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001531OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000331TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<2>, Int<1>>, Stride<Int<1>, Int<64>, Int<128>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 32, NUM_THREADS>::run(stensor20001527_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 512, NUM_THREADS>::run(stensor20001529_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20001526InputAtom::run(stensor20001526_async_copy_buf, dtensor10000330_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001526InputAtom::run(stensor20001526_ptr, dtensor10000330_tile_ptr + 256*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001526_ptr, stensor20001526_async_copy_buf);
    }
    {
      // OP type: tb_input_op
      STensor20001525InputAtom::run(stensor20001525_ptr, dtensor10000329_tile_ptr + 1024*for_idx, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<2>, Int<4>, Int<1>>, Stride<Int<1>, Int<8>, Int<32>>>, Layout<Shape<Int<2>, Int<4>, Int<1>>, Stride<Int<1>, Int<8>, Int<32>>>, NUM_THREADS>;
      Kernel::run(stensor20001527_ptr, stensor20001525_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, decltype(composition(Swizzle<5, 1, 7>{}, Layout<Shape<Int<256>, Int<2>, Int<1>>, Stride<Int<1>, Int<256>, Int<512>>>{})), Layout<Shape<Int<256>, Int<2>, Int<1>>, Stride<Int<1>, Int<256>, Int<512>>>, NUM_THREADS>;
      Kernel::run(stensor20001529_ptr, stensor20001526_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<2>, Int<4>, Int<1>>, Stride<Int<1>, Int<8>, Int<32>>>;
    using OutLayout = Layout<Shape<Int<2>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001528_ptr, stensor20001527_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = decltype(composition(Swizzle<5, 1, 7>{}, Layout<Shape<Int<2>, Int<256>, Int<1>>, Stride<Int<256>, Int<1>, Int<512>>>{}));
    using OutLayout = Layout<Shape<Int<2>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001530_ptr, stensor20001529_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<2>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using In1Layout = Layout<Shape<Int<2>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<2>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<128>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001531_ptr, stensor20001530_ptr, stensor20001528_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001531OutputAtom::run(dtensor10000331_tile_ptr, stensor20001531_ptr, thread_idx);
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
    half_t *dtensor10000329 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000330 = (half_t*)((char*)buf + 16384);
    half_t *dtensor10000326 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000327 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000328 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 57472;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 57472);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000329, dtensor10000330, dtensor10000326, dtensor10000327, dtensor10000328);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000331 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000329 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000330 = (half_t*)((char*)buf + 16384);
    dim3 grid_dim(2, 128, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 3328;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 3328);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000331, dtensor10000329, dtensor10000330);
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
