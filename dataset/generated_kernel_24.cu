#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000323_ptr, half_t* __restrict__ dtensor10000324_ptr, half_t const* __restrict__ dtensor10000320_ptr, half_t const* __restrict__ dtensor10000321_ptr, half_t const* __restrict__ dtensor10000322_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001485_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001483_ptr = (half_t*)(buf + 128);
  half_t *stensor20001481_ptr = (half_t*)(buf + 49280);
  half_t *stensor30001478_ptr = (half_t*)(buf + 41088);
  half_t *stensor20001478_ptr = (half_t*)(buf + 32896);
  half_t *stensor20001479_ptr = (half_t*)(buf + 24704);
  half_t *stensor30001479_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001482_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001477_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000320 -> stensor 20001477
  const half_t *dtensor10000320_tile_ptr = dtensor10000320_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000320TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001477InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000320TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000321 -> stensor 20001478
  const half_t *dtensor10000321_tile_ptr = dtensor10000321_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*1;
  using DTensor10000321TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001478InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000321TileLayout, NUM_THREADS>;
  half_t *stensor20001478_async_copy_buf = stensor30001478_ptr;
  // Copy for G->S: dtensor 10000322 -> stensor 20001479
  const half_t *dtensor10000322_tile_ptr = dtensor10000322_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*64;
  using DTensor10000322TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001479InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000322TileLayout, NUM_THREADS>;
  half_t *stensor20001479_async_copy_buf = stensor30001479_ptr;
  
  STensor20001477InputAtom::run(stensor20001477_ptr, dtensor10000320_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001485 -> dtensor 10000324
  half_t *dtensor10000324_tile_ptr = dtensor10000324_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*1 + blockIdx.z*64*1024;
  using DTensor10000324TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20001485OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000324TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20001483 -> dtensor 10000323
  half_t *dtensor10000323_tile_ptr = dtensor10000323_ptr  + blockIdx.x*1*1 + blockIdx.y*1*8 + blockIdx.z*64*128;
  using DTensor10000323TileLayout = Layout<Shape<Int<1>, Int<1>, Int<64>>, Stride<Int<1>, Int<8>, Int<128>>>;
  using STensor20001483OutputAtom = tb::OutputNonChunkedSyncCopy<half_t, DTensor10000323TileLayout, Layout<Shape<Int<1>, Int<1>, Int<64>>, Stride<Int<1>, Int<1>, Int<1>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001482_ptr, thread_idx);
  
  
  using Matmul20001481LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001481LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001481LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001481LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001481LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001481Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001481LayoutA, Matmul20001481LayoutB, Matmul20001481LayoutC, Matmul20001481LayoutAAligned, Matmul20001481LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001485LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001485LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001485LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001485LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001485LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001485Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001485LayoutA, Matmul20001485LayoutB, Matmul20001485LayoutC, Matmul20001485LayoutAAligned, Matmul20001485LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001485_accum = Matmul20001485Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001479InputAtom::run(stensor20001479_async_copy_buf, dtensor10000322_tile_ptr, thread_idx);
    STensor20001478InputAtom::run(stensor20001478_async_copy_buf, dtensor10000321_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001479InputAtom::run(stensor20001479_ptr, dtensor10000322_tile_ptr + 4096*(for_idx+1), thread_idx);
        STensor20001478InputAtom::run(stensor20001478_ptr, dtensor10000321_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001479_ptr, stensor20001479_async_copy_buf);
      SWAP(stensor20001478_ptr, stensor20001478_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001481Kernel::get_mma_rC(thread_idx);
      Matmul20001481Kernel::run(mma_rC, stensor20001477_ptr, stensor20001478_ptr, (char*)(buf+0), thread_idx);
      Matmul20001481Kernel::write_back_mma_rC(stensor20001481_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<1>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001482_ptr, stensor20001481_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001485Kernel::run(matmul_20001485_accum, stensor20001481_ptr, stensor20001479_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001485Kernel::write_back_mma_rC(stensor20001485_ptr, matmul_20001485_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<1>, Int<64>, Int<64>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<1>, Int<64>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001483_ptr, stensor20001482_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20001485OutputAtom::run(dtensor10000324_tile_ptr, stensor20001485_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001483OutputAtom::run(dtensor10000323_tile_ptr, stensor20001483_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000325_ptr, half_t const* __restrict__ dtensor10000323_ptr, half_t const* __restrict__ dtensor10000324_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001499_ptr = (half_t*)(buf + 400);
  half_t *stensor20001498_ptr = (half_t*)(buf + 144);
  half_t *stensor20001496_ptr = (half_t*)(buf + 128);
  half_t *stensor20001497_ptr = (half_t*)(buf + 4352);
  half_t *stensor20001495_ptr = (half_t*)(buf + 4288);
  half_t *stensor20001494_ptr = (half_t*)(buf + 192);
  half_t *stensor20001493_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000323 -> stensor 20001493
  const half_t *dtensor10000323_tile_ptr = dtensor10000323_ptr  + blockIdx.x*1*1 + blockIdx.y*2*128;
  using DTensor10000323TileLayout = Layout<Shape<Int<1>, Int<16>, Int<2>>, Stride<Int<1>, Int<8>, Int<128>>>;
  using STensor20001493InputAtom = tb::InputNonChunkedSyncCopy<half_t, Layout<Shape<Int<1>, Int<16>, Int<2>>, Stride<Int<1>, Int<1>, Int<16>>>, DTensor10000323TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000324 -> stensor 20001494
  const half_t *dtensor10000324_tile_ptr = dtensor10000324_ptr  + blockIdx.x*1*262144 + blockIdx.y*2*1024;
  using DTensor10000324TileLayout = Layout<Shape<Int<1024>, Int<2>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20001494InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<1024>, Int<2>, Int<1>>, Stride<Int<1>, Int<1024>, Int<2048>>>, DTensor10000324TileLayout, NUM_THREADS>;
  
  STensor20001493InputAtom::run(stensor20001493_ptr, dtensor10000323_tile_ptr, thread_idx);
  STensor20001494InputAtom::run(stensor20001494_ptr, dtensor10000324_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001499 -> dtensor 10000325
  half_t *dtensor10000325_tile_ptr = dtensor10000325_ptr  + blockIdx.x*1*16384 + blockIdx.y*2*64;
  using DTensor10000325TileLayout = Layout<Shape<Int<64>, Int<2>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001499OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000325TileLayout, Layout<Shape<Int<64>, Int<2>, Int<1>>, Stride<Int<1>, Int<64>, Int<128>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 32, NUM_THREADS>::run(stensor20001495_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001497_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<1>, Int<16>, Int<2>>, Stride<Int<1>, Int<1>, Int<16>>>, Layout<Shape<Int<1>, Int<16>, Int<2>>, Stride<Int<1>, Int<1>, Int<16>>>, NUM_THREADS>;
      Kernel::run(stensor20001495_ptr, stensor20001493_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<1024>, Int<2>, Int<1>>, Stride<Int<1>, Int<1024>, Int<1>>>, Layout<Shape<Int<1024>, Int<2>, Int<1>>, Stride<Int<1>, Int<1024>, Int<2048>>>, NUM_THREADS>;
      Kernel::run(stensor20001497_ptr, stensor20001494_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<1>, Int<16>, Int<2>>, Stride<Int<1>, Int<1>, Int<16>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<1>, Int<2>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001496_ptr, stensor20001495_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<1>, Int<1024>, Int<2>>, Stride<Int<1>, Int<1>, Int<1024>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001498_ptr, stensor20001497_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<2>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<128>, Int<1>, Int<64>>>;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001499_ptr, stensor20001498_ptr, stensor20001496_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001499OutputAtom::run(dtensor10000325_tile_ptr, stensor20001499_ptr, thread_idx);
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
    half_t *dtensor10000323 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000324 = (half_t*)((char*)buf + 65536);
    half_t *dtensor10000320 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000321 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000322 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 57472;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 57472);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000323, dtensor10000324, dtensor10000320, dtensor10000321, dtensor10000322);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000325 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000323 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000324 = (half_t*)((char*)buf + 65536);
    dim3 grid_dim(2, 128, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 8448;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 8448);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000325, dtensor10000323, dtensor10000324);
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
