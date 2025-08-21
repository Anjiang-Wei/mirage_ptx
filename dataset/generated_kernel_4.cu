#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000203_ptr, half_t* __restrict__ dtensor10000204_ptr, half_t const* __restrict__ dtensor10000200_ptr, half_t const* __restrict__ dtensor10000201_ptr, half_t const* __restrict__ dtensor10000202_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000863_ptr = (half_t*)(buf + 128);
  half_t *stensor20000860_ptr = (half_t*)(buf + 71808);
  half_t *stensor30000858_ptr = (half_t*)(buf + 55424);
  half_t *stensor20000858_ptr = (half_t*)(buf + 39040);
  half_t *stensor30000857_ptr = (half_t*)(buf + 22656);
  half_t *stensor20000857_ptr = (half_t*)(buf + 6272);
  half_t *stensor20000861_ptr = (half_t*)(buf + 2176);
  half_t *stensor20000856_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000200 -> stensor 20000856
  const half_t *dtensor10000200_tile_ptr = dtensor10000200_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000200TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000856InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000200TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000201 -> stensor 20000857
  const half_t *dtensor10000201_tile_ptr = dtensor10000201_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000201TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20000857InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000201TileLayout, NUM_THREADS>;
  half_t *stensor20000857_async_copy_buf = stensor30000857_ptr;
  // Copy for G->S: dtensor 10000202 -> stensor 20000858
  const half_t *dtensor10000202_tile_ptr = dtensor10000202_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000202TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20000858InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000202TileLayout, NUM_THREADS>;
  half_t *stensor20000858_async_copy_buf = stensor30000858_ptr;
  
  STensor20000856InputAtom::run(stensor20000856_ptr, dtensor10000200_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000861 -> dtensor 10000203
  half_t *dtensor10000203_tile_ptr = dtensor10000203_ptr  + blockIdx.x*1*262144 + blockIdx.y*128*256 + blockIdx.z*16*1;
  using DTensor10000203TileLayout = Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20000861OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000203TileLayout, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20000863 -> dtensor 10000204
  half_t *dtensor10000204_tile_ptr = dtensor10000204_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*256 + blockIdx.z*16*1;
  using DTensor10000204TileLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20000863OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000204TileLayout, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000861_ptr, thread_idx);
  
  
  using Matmul20000860LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000860LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000860LayoutC = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000860LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000860LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000860Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000860LayoutA, Matmul20000860LayoutB, Matmul20000860LayoutC, Matmul20000860LayoutAAligned, Matmul20000860LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20000863LayoutA = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000863LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000863LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20000863LayoutAAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000863LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000863Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000863LayoutA, Matmul20000863LayoutB, Matmul20000863LayoutC, Matmul20000863LayoutAAligned, Matmul20000863LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000863_accum = Matmul20000863Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20000858InputAtom::run(stensor20000858_async_copy_buf, dtensor10000202_tile_ptr, thread_idx);
    STensor20000857InputAtom::run(stensor20000857_async_copy_buf, dtensor10000201_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20000858InputAtom::run(stensor20000858_ptr, dtensor10000202_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20000857InputAtom::run(stensor20000857_ptr, dtensor10000201_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000858_ptr, stensor20000858_async_copy_buf);
      SWAP(stensor20000857_ptr, stensor20000857_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20000860Kernel::get_mma_rC(thread_idx);
      Matmul20000860Kernel::run(mma_rC, stensor20000856_ptr, stensor20000857_ptr, (char*)(buf+0), thread_idx);
      Matmul20000860Kernel::write_back_mma_rC(stensor20000860_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<128>, Int<1>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20000861_ptr, stensor20000860_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20000863Kernel::run(matmul_20000863_accum, stensor20000860_ptr, stensor20000858_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000863Kernel::write_back_mma_rC(stensor20000863_ptr, matmul_20000863_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000861OutputAtom::run(dtensor10000203_tile_ptr, stensor20000861_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20000863OutputAtom::run(dtensor10000204_tile_ptr, stensor20000863_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000205_ptr, half_t const* __restrict__ dtensor10000203_ptr, half_t const* __restrict__ dtensor10000204_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000877_ptr = (half_t*)(buf + 1168);
  half_t *stensor20000876_ptr = (half_t*)(buf + 144);
  half_t *stensor20000874_ptr = (half_t*)(buf + 128);
  half_t *stensor20000875_ptr = (half_t*)(buf + 41088);
  half_t *stensor20000873_ptr = (half_t*)(buf + 24704);
  half_t *stensor20000872_ptr = (half_t*)(buf + 16512);
  half_t *stensor20000871_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000203 -> stensor 20000871
  const half_t *dtensor10000203_tile_ptr = dtensor10000203_ptr  + blockIdx.x*1*262144 + blockIdx.y*8*1;
  using DTensor10000203TileLayout = Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20000871InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>, DTensor10000203TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000204 -> stensor 20000872
  const half_t *dtensor10000204_tile_ptr = dtensor10000204_ptr  + blockIdx.x*1*131072 + blockIdx.y*8*1;
  using DTensor10000204TileLayout = Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20000872InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>, DTensor10000204TileLayout, NUM_THREADS>;
  
  STensor20000871InputAtom::run(stensor20000871_ptr, dtensor10000203_tile_ptr, thread_idx);
  STensor20000872InputAtom::run(stensor20000872_ptr, dtensor10000204_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000877 -> dtensor 10000205
  half_t *dtensor10000205_tile_ptr = dtensor10000205_ptr  + blockIdx.x*1*16384 + blockIdx.y*8*64;
  using DTensor10000205TileLayout = Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000877OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000205TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<512>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 8192, NUM_THREADS>::run(stensor20000873_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20000875_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>, Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>, NUM_THREADS>;
      Kernel::run(stensor20000873_ptr, stensor20000871_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>, Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>, NUM_THREADS>;
      Kernel::run(stensor20000875_ptr, stensor20000872_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<8>, Int<1024>, Int<1>>, Stride<Int<1>, Int<8>, Int<8192>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000874_ptr, stensor20000873_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<8>, Int<512>, Int<1>>, Stride<Int<1>, Int<8>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000876_ptr, stensor20000875_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using In1Layout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<512>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000877_ptr, stensor20000876_ptr, stensor20000874_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000877OutputAtom::run(dtensor10000205_tile_ptr, stensor20000877_ptr, thread_idx);
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
    half_t *dtensor10000203 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000204 = (half_t*)((char*)buf + 1048576);
    half_t *dtensor10000200 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000201 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000202 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000203, dtensor10000204, dtensor10000200, dtensor10000201, dtensor10000202);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000205 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000203 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000204 = (half_t*)((char*)buf + 1048576);
    dim3 grid_dim(2, 32, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 49280;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 49280);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000205, dtensor10000203, dtensor10000204);
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
