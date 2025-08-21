#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000185_ptr, half_t* __restrict__ dtensor10000186_ptr, half_t const* __restrict__ dtensor10000182_ptr, half_t const* __restrict__ dtensor10000183_ptr, half_t const* __restrict__ dtensor10000184_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000773_ptr = (half_t*)(buf + 128);
  half_t *stensor20000770_ptr = (half_t*)(buf + 71808);
  half_t *stensor30000768_ptr = (half_t*)(buf + 55424);
  half_t *stensor20000768_ptr = (half_t*)(buf + 39040);
  half_t *stensor30000767_ptr = (half_t*)(buf + 22656);
  half_t *stensor20000767_ptr = (half_t*)(buf + 6272);
  half_t *stensor20000771_ptr = (half_t*)(buf + 2176);
  half_t *stensor20000766_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000182 -> stensor 20000766
  const half_t *dtensor10000182_tile_ptr = dtensor10000182_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000182TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000766InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000182TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000183 -> stensor 20000767
  const half_t *dtensor10000183_tile_ptr = dtensor10000183_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000183TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20000767InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000183TileLayout, NUM_THREADS>;
  half_t *stensor20000767_async_copy_buf = stensor30000767_ptr;
  // Copy for G->S: dtensor 10000184 -> stensor 20000768
  const half_t *dtensor10000184_tile_ptr = dtensor10000184_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000184TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20000768InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000184TileLayout, NUM_THREADS>;
  half_t *stensor20000768_async_copy_buf = stensor30000768_ptr;
  
  STensor20000766InputAtom::run(stensor20000766_ptr, dtensor10000182_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000771 -> dtensor 10000185
  half_t *dtensor10000185_tile_ptr = dtensor10000185_ptr  + blockIdx.x*1*262144 + blockIdx.y*128*256 + blockIdx.z*16*1;
  using DTensor10000185TileLayout = Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20000771OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000185TileLayout, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20000773 -> dtensor 10000186
  half_t *dtensor10000186_tile_ptr = dtensor10000186_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*256 + blockIdx.z*16*1;
  using DTensor10000186TileLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20000773OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000186TileLayout, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000771_ptr, thread_idx);
  
  
  using Matmul20000770LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000770LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000770LayoutC = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000770LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000770LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000770Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000770LayoutA, Matmul20000770LayoutB, Matmul20000770LayoutC, Matmul20000770LayoutAAligned, Matmul20000770LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20000773LayoutA = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000773LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000773LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20000773LayoutAAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000773LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000773Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000773LayoutA, Matmul20000773LayoutB, Matmul20000773LayoutC, Matmul20000773LayoutAAligned, Matmul20000773LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000773_accum = Matmul20000773Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20000768InputAtom::run(stensor20000768_async_copy_buf, dtensor10000184_tile_ptr, thread_idx);
    STensor20000767InputAtom::run(stensor20000767_async_copy_buf, dtensor10000183_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20000768InputAtom::run(stensor20000768_ptr, dtensor10000184_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20000767InputAtom::run(stensor20000767_ptr, dtensor10000183_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000768_ptr, stensor20000768_async_copy_buf);
      SWAP(stensor20000767_ptr, stensor20000767_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20000770Kernel::get_mma_rC(thread_idx);
      Matmul20000770Kernel::run(mma_rC, stensor20000766_ptr, stensor20000767_ptr, (char*)(buf+0), thread_idx);
      Matmul20000770Kernel::write_back_mma_rC(stensor20000770_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<128>, Int<1>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20000771_ptr, stensor20000770_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20000773Kernel::run(matmul_20000773_accum, stensor20000770_ptr, stensor20000768_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000773Kernel::write_back_mma_rC(stensor20000773_ptr, matmul_20000773_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000771OutputAtom::run(dtensor10000185_tile_ptr, stensor20000771_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20000773OutputAtom::run(dtensor10000186_tile_ptr, stensor20000773_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000187_ptr, half_t const* __restrict__ dtensor10000185_ptr, half_t const* __restrict__ dtensor10000186_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000787_ptr = (half_t*)(buf + 128);
  half_t *stensor20000786_ptr = (half_t*)(buf + 12448);
  half_t *stensor20000784_ptr = (half_t*)(buf + 12416);
  half_t *stensor30000781_ptr = (half_t*)(buf + 28800);
  half_t *stensor20000781_ptr = (half_t*)(buf + 20608);
  half_t *stensor20000782_ptr = (half_t*)(buf + 16512);
  half_t *stensor30000782_ptr = (half_t*)(buf + 12416);
  half_t *stensor20000785_ptr = (half_t*)(buf + 8320);
  half_t *stensor20000783_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000185 -> stensor 20000781
  const half_t *dtensor10000185_tile_ptr = dtensor10000185_ptr  + blockIdx.x*1*262144 + blockIdx.y*16*1;
  using DTensor10000185TileLayout = Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20000781InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>, DTensor10000185TileLayout, NUM_THREADS>;
  half_t *stensor20000781_async_copy_buf = stensor30000781_ptr;
  // Copy for G->S: dtensor 10000186 -> stensor 20000782
  const half_t *dtensor10000186_tile_ptr = dtensor10000186_ptr  + blockIdx.x*1*131072 + blockIdx.y*16*1;
  using DTensor10000186TileLayout = Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20000782InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, DTensor10000186TileLayout, NUM_THREADS>;
  half_t *stensor20000782_async_copy_buf = stensor30000782_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000787 -> dtensor 10000187
  half_t *dtensor10000187_tile_ptr = dtensor10000187_ptr  + blockIdx.x*1*16384 + blockIdx.y*16*64;
  using DTensor10000187TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000787OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000187TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20000783_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000785_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20000782InputAtom::run(stensor20000782_async_copy_buf, dtensor10000186_tile_ptr, thread_idx);
    STensor20000781InputAtom::run(stensor20000781_async_copy_buf, dtensor10000185_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20000782InputAtom::run(stensor20000782_ptr, dtensor10000186_tile_ptr + 32768*(for_idx+1), thread_idx);
        STensor20000781InputAtom::run(stensor20000781_ptr, dtensor10000185_tile_ptr + 65536*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000782_ptr, stensor20000782_async_copy_buf);
      SWAP(stensor20000781_ptr, stensor20000781_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>, Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>, NUM_THREADS>;
      Kernel::run(stensor20000783_ptr, stensor20000781_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, NUM_THREADS>;
      Kernel::run(stensor20000785_ptr, stensor20000782_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<16>, Int<256>, Int<1>>, Stride<Int<1>, Int<16>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000784_ptr, stensor20000783_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000786_ptr, stensor20000785_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using In1Layout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<1024>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000787_ptr, stensor20000786_ptr, stensor20000784_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000787OutputAtom::run(dtensor10000187_tile_ptr, stensor20000787_ptr, thread_idx);
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
    half_t *dtensor10000185 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000186 = (half_t*)((char*)buf + 1048576);
    half_t *dtensor10000182 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000183 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000184 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000185, dtensor10000186, dtensor10000182, dtensor10000183, dtensor10000184);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000187 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000185 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000186 = (half_t*)((char*)buf + 1048576);
    dim3 grid_dim(2, 16, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 36992;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 36992);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000187, dtensor10000185, dtensor10000186);
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
