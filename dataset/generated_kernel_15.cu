#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000269_ptr, half_t* __restrict__ dtensor10000270_ptr, half_t const* __restrict__ dtensor10000266_ptr, half_t const* __restrict__ dtensor10000267_ptr, half_t const* __restrict__ dtensor10000268_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001209_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001207_ptr = (half_t*)(buf + 128);
  half_t *stensor20001205_ptr = (half_t*)(buf + 71808);
  half_t *stensor30001202_ptr = (half_t*)(buf + 55424);
  half_t *stensor20001202_ptr = (half_t*)(buf + 39040);
  half_t *stensor20001203_ptr = (half_t*)(buf + 22656);
  half_t *stensor30001203_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001206_ptr = (half_t*)(buf + 2176);
  half_t *stensor20001201_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000266 -> stensor 20001201
  const half_t *dtensor10000266_tile_ptr = dtensor10000266_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000266TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001201InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000266TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000267 -> stensor 20001202
  const half_t *dtensor10000267_tile_ptr = dtensor10000267_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000267TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001202InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000267TileLayout, NUM_THREADS>;
  half_t *stensor20001202_async_copy_buf = stensor30001202_ptr;
  // Copy for G->S: dtensor 10000268 -> stensor 20001203
  const half_t *dtensor10000268_tile_ptr = dtensor10000268_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000268TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001203InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000268TileLayout, NUM_THREADS>;
  half_t *stensor20001203_async_copy_buf = stensor30001203_ptr;
  
  STensor20001201InputAtom::run(stensor20001201_ptr, dtensor10000266_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001209 -> dtensor 10000270
  half_t *dtensor10000270_tile_ptr = dtensor10000270_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*256 + blockIdx.z*16*1;
  using DTensor10000270TileLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20001209OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000270TileLayout, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20001207 -> dtensor 10000269
  half_t *dtensor10000269_tile_ptr = dtensor10000269_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*256 + blockIdx.z*16*1;
  using DTensor10000269TileLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20001207OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000269TileLayout, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001206_ptr, thread_idx);
  
  
  using Matmul20001205LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001205LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001205LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001205LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001205LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001205Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001205LayoutA, Matmul20001205LayoutB, Matmul20001205LayoutC, Matmul20001205LayoutAAligned, Matmul20001205LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001209LayoutA = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001209LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001209LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001209LayoutAAligned = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001209LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001209Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001209LayoutA, Matmul20001209LayoutB, Matmul20001209LayoutC, Matmul20001209LayoutAAligned, Matmul20001209LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001209_accum = Matmul20001209Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001203InputAtom::run(stensor20001203_async_copy_buf, dtensor10000268_tile_ptr, thread_idx);
    STensor20001202InputAtom::run(stensor20001202_async_copy_buf, dtensor10000267_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001203InputAtom::run(stensor20001203_ptr, dtensor10000268_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20001202InputAtom::run(stensor20001202_ptr, dtensor10000267_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001203_ptr, stensor20001203_async_copy_buf);
      SWAP(stensor20001202_ptr, stensor20001202_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001205Kernel::get_mma_rC(thread_idx);
      Matmul20001205Kernel::run(mma_rC, stensor20001201_ptr, stensor20001202_ptr, (char*)(buf+0), thread_idx);
      Matmul20001205Kernel::write_back_mma_rC(stensor20001205_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001206_ptr, stensor20001205_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001209Kernel::run(matmul_20001209_accum, stensor20001205_ptr, stensor20001203_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001209Kernel::write_back_mma_rC(stensor20001209_ptr, matmul_20001209_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001207_ptr, stensor20001206_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20001209OutputAtom::run(dtensor10000270_tile_ptr, stensor20001209_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001207OutputAtom::run(dtensor10000269_tile_ptr, stensor20001207_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000271_ptr, half_t const* __restrict__ dtensor10000269_ptr, half_t const* __restrict__ dtensor10000270_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001223_ptr = (half_t*)(buf + 128);
  half_t *stensor20001222_ptr = (half_t*)(buf + 4240);
  half_t *stensor20001220_ptr = (half_t*)(buf + 4224);
  half_t *stensor30001217_ptr = (half_t*)(buf + 10368);
  half_t *stensor20001217_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001218_ptr = (half_t*)(buf + 6272);
  half_t *stensor30001218_ptr = (half_t*)(buf + 4224);
  half_t *stensor20001221_ptr = (half_t*)(buf + 2176);
  half_t *stensor20001219_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000269 -> stensor 20001217
  const half_t *dtensor10000269_tile_ptr = dtensor10000269_ptr  + blockIdx.x*1*131072 + blockIdx.y*8*1;
  using DTensor10000269TileLayout = Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20001217InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>, DTensor10000269TileLayout, NUM_THREADS>;
  half_t *stensor20001217_async_copy_buf = stensor30001217_ptr;
  // Copy for G->S: dtensor 10000270 -> stensor 20001218
  const half_t *dtensor10000270_tile_ptr = dtensor10000270_ptr  + blockIdx.x*1*131072 + blockIdx.y*8*1;
  using DTensor10000270TileLayout = Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20001218InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>, DTensor10000270TileLayout, NUM_THREADS>;
  half_t *stensor20001218_async_copy_buf = stensor30001218_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001223 -> dtensor 10000271
  half_t *dtensor10000271_tile_ptr = dtensor10000271_ptr  + blockIdx.x*1*16384 + blockIdx.y*8*64;
  using DTensor10000271TileLayout = Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001223OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000271TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<512>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 1024, NUM_THREADS>::run(stensor20001219_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 1024, NUM_THREADS>::run(stensor20001221_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20001218InputAtom::run(stensor20001218_async_copy_buf, dtensor10000270_tile_ptr, thread_idx);
    STensor20001217InputAtom::run(stensor20001217_async_copy_buf, dtensor10000269_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001218InputAtom::run(stensor20001218_ptr, dtensor10000270_tile_ptr + 32768*(for_idx+1), thread_idx);
        STensor20001217InputAtom::run(stensor20001217_ptr, dtensor10000269_tile_ptr + 32768*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001218_ptr, stensor20001218_async_copy_buf);
      SWAP(stensor20001217_ptr, stensor20001217_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>, Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>, NUM_THREADS>;
      Kernel::run(stensor20001219_ptr, stensor20001217_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>, Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>, NUM_THREADS>;
      Kernel::run(stensor20001221_ptr, stensor20001218_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001220_ptr, stensor20001219_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001222_ptr, stensor20001221_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using In1Layout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<512>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001223_ptr, stensor20001222_ptr, stensor20001220_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001223OutputAtom::run(dtensor10000271_tile_ptr, stensor20001223_ptr, thread_idx);
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
    half_t *dtensor10000269 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000270 = (half_t*)((char*)buf + 524288);
    half_t *dtensor10000266 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000267 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000268 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000269, dtensor10000270, dtensor10000266, dtensor10000267, dtensor10000268);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000271 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000269 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000270 = (half_t*)((char*)buf + 524288);
    dim3 grid_dim(2, 32, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 12416;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 12416);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000271, dtensor10000269, dtensor10000270);
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
