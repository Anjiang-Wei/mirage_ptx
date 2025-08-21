#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000251_ptr, half_t* __restrict__ dtensor10000252_ptr, half_t const* __restrict__ dtensor10000248_ptr, half_t const* __restrict__ dtensor10000249_ptr, half_t const* __restrict__ dtensor10000250_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001113_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001111_ptr = (half_t*)(buf + 128);
  half_t *stensor20001109_ptr = (half_t*)(buf + 71808);
  half_t *stensor30001106_ptr = (half_t*)(buf + 55424);
  half_t *stensor20001106_ptr = (half_t*)(buf + 39040);
  half_t *stensor20001107_ptr = (half_t*)(buf + 22656);
  half_t *stensor30001107_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001110_ptr = (half_t*)(buf + 2176);
  half_t *stensor20001105_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000248 -> stensor 20001105
  const half_t *dtensor10000248_tile_ptr = dtensor10000248_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000248TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001105InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000248TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000249 -> stensor 20001106
  const half_t *dtensor10000249_tile_ptr = dtensor10000249_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000249TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001106InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000249TileLayout, NUM_THREADS>;
  half_t *stensor20001106_async_copy_buf = stensor30001106_ptr;
  // Copy for G->S: dtensor 10000250 -> stensor 20001107
  const half_t *dtensor10000250_tile_ptr = dtensor10000250_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000250TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001107InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000250TileLayout, NUM_THREADS>;
  half_t *stensor20001107_async_copy_buf = stensor30001107_ptr;
  
  STensor20001105InputAtom::run(stensor20001105_ptr, dtensor10000248_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001113 -> dtensor 10000252
  half_t *dtensor10000252_tile_ptr = dtensor10000252_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*1 + blockIdx.z*16*512;
  using DTensor10000252TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20001113OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000252TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20001111 -> dtensor 10000251
  half_t *dtensor10000251_tile_ptr = dtensor10000251_ptr  + blockIdx.x*1*1 + blockIdx.y*64*8 + blockIdx.z*16*4096;
  using DTensor10000251TileLayout = Layout<Shape<Int<1>, Int<64>, Int<16>>, Stride<Int<1>, Int<8>, Int<4096>>>;
  using STensor20001111OutputAtom = tb::OutputNonChunkedSyncCopy<half_t, DTensor10000251TileLayout, Layout<Shape<Int<1>, Int<64>, Int<16>>, Stride<Int<1>, Int<1>, Int<64>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001110_ptr, thread_idx);
  
  
  using Matmul20001109LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001109LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001109LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001109LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001109LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001109Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001109LayoutA, Matmul20001109LayoutB, Matmul20001109LayoutC, Matmul20001109LayoutAAligned, Matmul20001109LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001113LayoutA = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001113LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001113LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001113LayoutAAligned = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001113LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001113Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001113LayoutA, Matmul20001113LayoutB, Matmul20001113LayoutC, Matmul20001113LayoutAAligned, Matmul20001113LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001113_accum = Matmul20001113Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001107InputAtom::run(stensor20001107_async_copy_buf, dtensor10000250_tile_ptr, thread_idx);
    STensor20001106InputAtom::run(stensor20001106_async_copy_buf, dtensor10000249_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001107InputAtom::run(stensor20001107_ptr, dtensor10000250_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20001106InputAtom::run(stensor20001106_ptr, dtensor10000249_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001107_ptr, stensor20001107_async_copy_buf);
      SWAP(stensor20001106_ptr, stensor20001106_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001109Kernel::get_mma_rC(thread_idx);
      Matmul20001109Kernel::run(mma_rC, stensor20001105_ptr, stensor20001106_ptr, (char*)(buf+0), thread_idx);
      Matmul20001109Kernel::write_back_mma_rC(stensor20001109_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<128>, Int<1>, Int<1>>>, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001110_ptr, stensor20001109_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001113Kernel::run(matmul_20001113_accum, stensor20001109_ptr, stensor20001107_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001113Kernel::write_back_mma_rC(stensor20001113_ptr, matmul_20001113_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<1>, Int<128>, Int<16>>, Stride<Int<1>, Int<1>, Int<128>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<16>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001111_ptr, stensor20001110_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20001113OutputAtom::run(dtensor10000252_tile_ptr, stensor20001113_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001111OutputAtom::run(dtensor10000251_tile_ptr, stensor20001111_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000253_ptr, half_t const* __restrict__ dtensor10000251_ptr, half_t const* __restrict__ dtensor10000252_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001127_ptr = (half_t*)(buf + 656);
  half_t *stensor20001126_ptr = (half_t*)(buf + 144);
  half_t *stensor20001124_ptr = (half_t*)(buf + 128);
  half_t *stensor20001125_ptr = (half_t*)(buf + 12416);
  half_t *stensor20001123_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001122_ptr = (half_t*)(buf + 4224);
  half_t *stensor20001121_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000251 -> stensor 20001121
  const half_t *dtensor10000251_tile_ptr = dtensor10000251_ptr  + blockIdx.x*1*1 + blockIdx.y*4*4096;
  using DTensor10000251TileLayout = Layout<Shape<Int<1>, Int<512>, Int<4>>, Stride<Int<1>, Int<8>, Int<4096>>>;
  using STensor20001121InputAtom = tb::InputNonChunkedSyncCopy<half_t, Layout<Shape<Int<1>, Int<512>, Int<4>>, Stride<Int<1>, Int<1>, Int<512>>>, DTensor10000251TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000252 -> stensor 20001122
  const half_t *dtensor10000252_tile_ptr = dtensor10000252_ptr  + blockIdx.x*1*131072 + blockIdx.y*4*512;
  using DTensor10000252TileLayout = Layout<Shape<Int<512>, Int<4>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20001122InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<512>, Int<4>, Int<1>>, Stride<Int<1>, Int<512>, Int<2048>>>, DTensor10000252TileLayout, NUM_THREADS>;
  
  STensor20001121InputAtom::run(stensor20001121_ptr, dtensor10000251_tile_ptr, thread_idx);
  STensor20001122InputAtom::run(stensor20001122_ptr, dtensor10000252_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001127 -> dtensor 10000253
  half_t *dtensor10000253_tile_ptr = dtensor10000253_ptr  + blockIdx.x*1*16384 + blockIdx.y*4*64;
  using DTensor10000253TileLayout = Layout<Shape<Int<64>, Int<4>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001127OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000253TileLayout, Layout<Shape<Int<64>, Int<4>, Int<1>>, Stride<Int<1>, Int<64>, Int<256>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001123_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001125_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<1>, Int<512>, Int<4>>, Stride<Int<1>, Int<1>, Int<512>>>, Layout<Shape<Int<1>, Int<512>, Int<4>>, Stride<Int<1>, Int<1>, Int<512>>>, NUM_THREADS>;
      Kernel::run(stensor20001123_ptr, stensor20001121_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<1>, Int<512>, Int<4>>, Stride<Int<1>, Int<1>, Int<512>>>, Layout<Shape<Int<1>, Int<512>, Int<4>>, Stride<Int<2048>, Int<1>, Int<512>>>, NUM_THREADS>;
      Kernel::run(stensor20001125_ptr, stensor20001122_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<1>, Int<512>, Int<4>>, Stride<Int<1>, Int<1>, Int<512>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<1>, Int<4>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001124_ptr, stensor20001123_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<1>, Int<512>, Int<4>>, Stride<Int<1>, Int<1>, Int<512>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<4>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001126_ptr, stensor20001125_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<1>, Int<64>, Int<4>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<4>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<4>>, Stride<Int<256>, Int<1>, Int<64>>>;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001127_ptr, stensor20001126_ptr, stensor20001124_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001127OutputAtom::run(dtensor10000253_tile_ptr, stensor20001127_ptr, thread_idx);
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
    half_t *dtensor10000251 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000252 = (half_t*)((char*)buf + 2097152);
    half_t *dtensor10000248 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000249 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000250 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000251, dtensor10000252, dtensor10000248, dtensor10000249, dtensor10000250);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000253 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000251 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000252 = (half_t*)((char*)buf + 2097152);
    dim3 grid_dim(2, 64, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 16512;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 16512);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000253, dtensor10000251, dtensor10000252);
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
