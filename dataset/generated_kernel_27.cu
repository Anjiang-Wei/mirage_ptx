#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000341_ptr, half_t* __restrict__ dtensor10000342_ptr, half_t const* __restrict__ dtensor10000338_ptr, half_t const* __restrict__ dtensor10000339_ptr, half_t const* __restrict__ dtensor10000340_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001581_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001579_ptr = (half_t*)(buf + 128);
  half_t *stensor20001577_ptr = (half_t*)(buf + 49280);
  half_t *stensor30001574_ptr = (half_t*)(buf + 41088);
  half_t *stensor20001574_ptr = (half_t*)(buf + 32896);
  half_t *stensor20001575_ptr = (half_t*)(buf + 24704);
  half_t *stensor30001575_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001578_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001573_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000338 -> stensor 20001573
  const half_t *dtensor10000338_tile_ptr = dtensor10000338_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000338TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001573InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000338TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000339 -> stensor 20001574
  const half_t *dtensor10000339_tile_ptr = dtensor10000339_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*1;
  using DTensor10000339TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001574InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000339TileLayout, NUM_THREADS>;
  half_t *stensor20001574_async_copy_buf = stensor30001574_ptr;
  // Copy for G->S: dtensor 10000340 -> stensor 20001575
  const half_t *dtensor10000340_tile_ptr = dtensor10000340_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*64;
  using DTensor10000340TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001575InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000340TileLayout, NUM_THREADS>;
  half_t *stensor20001575_async_copy_buf = stensor30001575_ptr;
  
  STensor20001573InputAtom::run(stensor20001573_ptr, dtensor10000338_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001581 -> dtensor 10000342
  half_t *dtensor10000342_tile_ptr = dtensor10000342_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*256 + blockIdx.z*64*1;
  using DTensor10000342TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001581OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000342TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20001579 -> dtensor 10000341
  half_t *dtensor10000341_tile_ptr = dtensor10000341_ptr  + blockIdx.x*1*4096 + blockIdx.y*1*256 + blockIdx.z*64*1;
  using DTensor10000341TileLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20001579OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000341TileLayout, Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001578_ptr, thread_idx);
  
  
  using Matmul20001577LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001577LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001577LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001577LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001577LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001577Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001577LayoutA, Matmul20001577LayoutB, Matmul20001577LayoutC, Matmul20001577LayoutAAligned, Matmul20001577LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001581LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001581LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001581LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001581LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001581LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001581Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001581LayoutA, Matmul20001581LayoutB, Matmul20001581LayoutC, Matmul20001581LayoutAAligned, Matmul20001581LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001581_accum = Matmul20001581Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001575InputAtom::run(stensor20001575_async_copy_buf, dtensor10000340_tile_ptr, thread_idx);
    STensor20001574InputAtom::run(stensor20001574_async_copy_buf, dtensor10000339_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001575InputAtom::run(stensor20001575_ptr, dtensor10000340_tile_ptr + 4096*(for_idx+1), thread_idx);
        STensor20001574InputAtom::run(stensor20001574_ptr, dtensor10000339_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001575_ptr, stensor20001575_async_copy_buf);
      SWAP(stensor20001574_ptr, stensor20001574_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001577Kernel::get_mma_rC(thread_idx);
      Matmul20001577Kernel::run(mma_rC, stensor20001573_ptr, stensor20001574_ptr, (char*)(buf+0), thread_idx);
      Matmul20001577Kernel::write_back_mma_rC(stensor20001577_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001578_ptr, stensor20001577_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001581Kernel::run(matmul_20001581_accum, stensor20001577_ptr, stensor20001575_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001581Kernel::write_back_mma_rC(stensor20001581_ptr, matmul_20001581_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001579_ptr, stensor20001578_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20001581OutputAtom::run(dtensor10000342_tile_ptr, stensor20001581_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001579OutputAtom::run(dtensor10000341_tile_ptr, stensor20001579_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000343_ptr, half_t const* __restrict__ dtensor10000341_ptr, half_t const* __restrict__ dtensor10000342_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001595_ptr = (half_t*)(buf + 128);
  half_t *stensor20001594_ptr = (half_t*)(buf + 4304);
  half_t *stensor20001592_ptr = (half_t*)(buf + 4288);
  half_t *stensor30001589_ptr = (half_t*)(buf + 12544);
  half_t *stensor20001589_ptr = (half_t*)(buf + 12480);
  half_t *stensor20001590_ptr = (half_t*)(buf + 8384);
  half_t *stensor30001590_ptr = (half_t*)(buf + 4288);
  half_t *stensor20001593_ptr = (half_t*)(buf + 192);
  half_t *stensor20001591_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000341 -> stensor 20001589
  const half_t *dtensor10000341_tile_ptr = dtensor10000341_ptr  + blockIdx.x*1*4096 + blockIdx.y*8*1;
  using DTensor10000341TileLayout = Layout<Shape<Int<8>, Int<4>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20001589InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<8>, Int<4>, Int<1>>, Stride<Int<1>, Int<8>, Int<32>>>, DTensor10000341TileLayout, NUM_THREADS>;
  half_t *stensor20001589_async_copy_buf = stensor30001589_ptr;
  // Copy for G->S: dtensor 10000342 -> stensor 20001590
  const half_t *dtensor10000342_tile_ptr = dtensor10000342_ptr  + blockIdx.x*1*262144 + blockIdx.y*8*1;
  using DTensor10000342TileLayout = Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001590InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>, DTensor10000342TileLayout, NUM_THREADS>;
  half_t *stensor20001590_async_copy_buf = stensor30001590_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001595 -> dtensor 10000343
  half_t *dtensor10000343_tile_ptr = dtensor10000343_ptr  + blockIdx.x*1*16384 + blockIdx.y*8*64;
  using DTensor10000343TileLayout = Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001595OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000343TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<512>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 32, NUM_THREADS>::run(stensor20001591_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001593_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20001590InputAtom::run(stensor20001590_async_copy_buf, dtensor10000342_tile_ptr, thread_idx);
    STensor20001589InputAtom::run(stensor20001589_async_copy_buf, dtensor10000341_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001590InputAtom::run(stensor20001590_ptr, dtensor10000342_tile_ptr + 65536*(for_idx+1), thread_idx);
        STensor20001589InputAtom::run(stensor20001589_ptr, dtensor10000341_tile_ptr + 1024*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001590_ptr, stensor20001590_async_copy_buf);
      SWAP(stensor20001589_ptr, stensor20001589_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<4>, Int<1>>, Stride<Int<1>, Int<8>, Int<32>>>, Layout<Shape<Int<8>, Int<4>, Int<1>>, Stride<Int<1>, Int<8>, Int<32>>>, NUM_THREADS>;
      Kernel::run(stensor20001591_ptr, stensor20001589_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>, Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>, NUM_THREADS>;
      Kernel::run(stensor20001593_ptr, stensor20001590_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<8>, Int<4>, Int<1>>, Stride<Int<1>, Int<8>, Int<32>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001592_ptr, stensor20001591_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001594_ptr, stensor20001593_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using In1Layout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<512>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001595_ptr, stensor20001594_ptr, stensor20001592_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001595OutputAtom::run(dtensor10000343_tile_ptr, stensor20001595_ptr, thread_idx);
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
    half_t *dtensor10000341 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000342 = (half_t*)((char*)buf + 16384);
    half_t *dtensor10000338 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000339 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000340 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 57472;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 57472);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000341, dtensor10000342, dtensor10000338, dtensor10000339, dtensor10000340);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000343 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000341 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000342 = (half_t*)((char*)buf + 16384);
    dim3 grid_dim(2, 32, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 12608;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 12608);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000343, dtensor10000341, dtensor10000342);
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
