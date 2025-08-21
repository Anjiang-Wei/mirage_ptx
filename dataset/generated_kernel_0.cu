#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000179_ptr, half_t* __restrict__ dtensor10000180_ptr, half_t const* __restrict__ dtensor10000176_ptr, half_t const* __restrict__ dtensor10000177_ptr, half_t const* __restrict__ dtensor10000178_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000743_ptr = (half_t*)(buf + 128);
  half_t *stensor20000740_ptr = (half_t*)(buf + 71808);
  half_t *stensor30000738_ptr = (half_t*)(buf + 55424);
  half_t *stensor20000738_ptr = (half_t*)(buf + 39040);
  half_t *stensor30000737_ptr = (half_t*)(buf + 22656);
  half_t *stensor20000737_ptr = (half_t*)(buf + 6272);
  half_t *stensor20000741_ptr = (half_t*)(buf + 2176);
  half_t *stensor20000736_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000176 -> stensor 20000736
  const half_t *dtensor10000176_tile_ptr = dtensor10000176_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000176TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000736InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000176TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000177 -> stensor 20000737
  const half_t *dtensor10000177_tile_ptr = dtensor10000177_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000177TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20000737InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000177TileLayout, NUM_THREADS>;
  half_t *stensor20000737_async_copy_buf = stensor30000737_ptr;
  // Copy for G->S: dtensor 10000178 -> stensor 20000738
  const half_t *dtensor10000178_tile_ptr = dtensor10000178_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000178TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20000738InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000178TileLayout, NUM_THREADS>;
  half_t *stensor20000738_async_copy_buf = stensor30000738_ptr;
  
  STensor20000736InputAtom::run(stensor20000736_ptr, dtensor10000176_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000741 -> dtensor 10000179
  half_t *dtensor10000179_tile_ptr = dtensor10000179_ptr  + blockIdx.x*1*262144 + blockIdx.y*128*256 + blockIdx.z*16*1;
  using DTensor10000179TileLayout = Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20000741OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000179TileLayout, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20000743 -> dtensor 10000180
  half_t *dtensor10000180_tile_ptr = dtensor10000180_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*256 + blockIdx.z*16*1;
  using DTensor10000180TileLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20000743OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000180TileLayout, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000741_ptr, thread_idx);
  
  
  using Matmul20000740LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000740LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000740LayoutC = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000740LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000740LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000740Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000740LayoutA, Matmul20000740LayoutB, Matmul20000740LayoutC, Matmul20000740LayoutAAligned, Matmul20000740LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20000743LayoutA = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000743LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000743LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20000743LayoutAAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000743LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000743Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000743LayoutA, Matmul20000743LayoutB, Matmul20000743LayoutC, Matmul20000743LayoutAAligned, Matmul20000743LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000743_accum = Matmul20000743Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20000738InputAtom::run(stensor20000738_async_copy_buf, dtensor10000178_tile_ptr, thread_idx);
    STensor20000737InputAtom::run(stensor20000737_async_copy_buf, dtensor10000177_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20000738InputAtom::run(stensor20000738_ptr, dtensor10000178_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20000737InputAtom::run(stensor20000737_ptr, dtensor10000177_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000738_ptr, stensor20000738_async_copy_buf);
      SWAP(stensor20000737_ptr, stensor20000737_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20000740Kernel::get_mma_rC(thread_idx);
      Matmul20000740Kernel::run(mma_rC, stensor20000736_ptr, stensor20000737_ptr, (char*)(buf+0), thread_idx);
      Matmul20000740Kernel::write_back_mma_rC(stensor20000740_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<128>, Int<1>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20000741_ptr, stensor20000740_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20000743Kernel::run(matmul_20000743_accum, stensor20000740_ptr, stensor20000738_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000743Kernel::write_back_mma_rC(stensor20000743_ptr, matmul_20000743_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000741OutputAtom::run(dtensor10000179_tile_ptr, stensor20000741_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20000743OutputAtom::run(dtensor10000180_tile_ptr, stensor20000743_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000181_ptr, half_t const* __restrict__ dtensor10000179_ptr, half_t const* __restrict__ dtensor10000180_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000757_ptr = (half_t*)(buf + 2208);
  half_t *stensor20000756_ptr = (half_t*)(buf + 160);
  half_t *stensor20000754_ptr = (half_t*)(buf + 128);
  half_t *stensor20000755_ptr = (half_t*)(buf + 82048);
  half_t *stensor20000753_ptr = (half_t*)(buf + 49280);
  half_t *stensor20000752_ptr = (half_t*)(buf + 32896);
  half_t *stensor20000751_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000179 -> stensor 20000751
  const half_t *dtensor10000179_tile_ptr = dtensor10000179_ptr  + blockIdx.x*1*262144 + blockIdx.y*16*1;
  using DTensor10000179TileLayout = Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20000751InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, DTensor10000179TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000180 -> stensor 20000752
  const half_t *dtensor10000180_tile_ptr = dtensor10000180_ptr  + blockIdx.x*1*131072 + blockIdx.y*16*1;
  using DTensor10000180TileLayout = Layout<Shape<Int<16>, Int<512>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20000752InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<16>, Int<512>, Int<1>>, Stride<Int<1>, Int<16>, Int<8192>>>, DTensor10000180TileLayout, NUM_THREADS>;
  
  STensor20000751InputAtom::run(stensor20000751_ptr, dtensor10000179_tile_ptr, thread_idx);
  STensor20000752InputAtom::run(stensor20000752_ptr, dtensor10000180_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000757 -> dtensor 10000181
  half_t *dtensor10000181_tile_ptr = dtensor10000181_ptr  + blockIdx.x*1*16384 + blockIdx.y*16*64;
  using DTensor10000181TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000757OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000181TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 16384, NUM_THREADS>::run(stensor20000753_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 8192, NUM_THREADS>::run(stensor20000755_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, NUM_THREADS>;
      Kernel::run(stensor20000753_ptr, stensor20000751_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<512>, Int<1>>, Stride<Int<1>, Int<16>, Int<8192>>>, Layout<Shape<Int<16>, Int<512>, Int<1>>, Stride<Int<1>, Int<16>, Int<8192>>>, NUM_THREADS>;
      Kernel::run(stensor20000755_ptr, stensor20000752_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000754_ptr, stensor20000753_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<16>, Int<512>, Int<1>>, Stride<Int<1>, Int<16>, Int<8192>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000756_ptr, stensor20000755_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using In1Layout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<1024>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000757_ptr, stensor20000756_ptr, stensor20000754_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000757OutputAtom::run(dtensor10000181_tile_ptr, stensor20000757_ptr, thread_idx);
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
    half_t *dtensor10000179 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000180 = (half_t*)((char*)buf + 1048576);
    half_t *dtensor10000176 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000177 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000178 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000179, dtensor10000180, dtensor10000176, dtensor10000177, dtensor10000178);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000181 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000179 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000180 = (half_t*)((char*)buf + 1048576);
    dim3 grid_dim(2, 16, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 98432;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 98432);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000181, dtensor10000179, dtensor10000180);
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
