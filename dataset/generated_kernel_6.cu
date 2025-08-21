#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000215_ptr, half_t* __restrict__ dtensor10000216_ptr, half_t const* __restrict__ dtensor10000212_ptr, half_t const* __restrict__ dtensor10000213_ptr, half_t const* __restrict__ dtensor10000214_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000923_ptr = (half_t*)(buf + 128);
  half_t *stensor20000920_ptr = (half_t*)(buf + 71808);
  half_t *stensor30000918_ptr = (half_t*)(buf + 55424);
  half_t *stensor20000918_ptr = (half_t*)(buf + 39040);
  half_t *stensor30000917_ptr = (half_t*)(buf + 22656);
  half_t *stensor20000917_ptr = (half_t*)(buf + 6272);
  half_t *stensor20000921_ptr = (half_t*)(buf + 2176);
  half_t *stensor20000916_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000212 -> stensor 20000916
  const half_t *dtensor10000212_tile_ptr = dtensor10000212_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000212TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000916InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000212TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000213 -> stensor 20000917
  const half_t *dtensor10000213_tile_ptr = dtensor10000213_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000213TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20000917InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000213TileLayout, NUM_THREADS>;
  half_t *stensor20000917_async_copy_buf = stensor30000917_ptr;
  // Copy for G->S: dtensor 10000214 -> stensor 20000918
  const half_t *dtensor10000214_tile_ptr = dtensor10000214_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000214TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20000918InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000214TileLayout, NUM_THREADS>;
  half_t *stensor20000918_async_copy_buf = stensor30000918_ptr;
  
  STensor20000916InputAtom::run(stensor20000916_ptr, dtensor10000212_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000921 -> dtensor 10000215
  half_t *dtensor10000215_tile_ptr = dtensor10000215_ptr  + blockIdx.x*1*1 + blockIdx.y*128*8 + blockIdx.z*16*8192;
  using DTensor10000215TileLayout = Layout<Shape<Int<1>, Int<128>, Int<16>>, Stride<Int<1>, Int<8>, Int<8192>>>;
  using STensor20000921OutputAtom = tb::OutputNonChunkedSyncCopy<half_t, DTensor10000215TileLayout, Layout<Shape<Int<1>, Int<128>, Int<16>>, Stride<Int<1>, Int<1>, Int<128>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20000923 -> dtensor 10000216
  half_t *dtensor10000216_tile_ptr = dtensor10000216_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*1 + blockIdx.z*16*512;
  using DTensor10000216TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20000923OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000216TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000921_ptr, thread_idx);
  
  
  using Matmul20000920LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000920LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000920LayoutC = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000920LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000920LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000920Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000920LayoutA, Matmul20000920LayoutB, Matmul20000920LayoutC, Matmul20000920LayoutAAligned, Matmul20000920LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20000923LayoutA = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000923LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000923LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000923LayoutAAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000923LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000923Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000923LayoutA, Matmul20000923LayoutB, Matmul20000923LayoutC, Matmul20000923LayoutAAligned, Matmul20000923LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000923_accum = Matmul20000923Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20000918InputAtom::run(stensor20000918_async_copy_buf, dtensor10000214_tile_ptr, thread_idx);
    STensor20000917InputAtom::run(stensor20000917_async_copy_buf, dtensor10000213_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20000918InputAtom::run(stensor20000918_ptr, dtensor10000214_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20000917InputAtom::run(stensor20000917_ptr, dtensor10000213_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000918_ptr, stensor20000918_async_copy_buf);
      SWAP(stensor20000917_ptr, stensor20000917_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20000920Kernel::get_mma_rC(thread_idx);
      Matmul20000920Kernel::run(mma_rC, stensor20000916_ptr, stensor20000917_ptr, (char*)(buf+0), thread_idx);
      Matmul20000920Kernel::write_back_mma_rC(stensor20000920_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<128>, Int<16>, Int<1>>, Stride<Int<1>, Int<128>, Int<1>>>, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>, Int<1>>, Stride<Int<1>, Int<128>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20000921_ptr, stensor20000920_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20000923Kernel::run(matmul_20000923_accum, stensor20000920_ptr, stensor20000918_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000923Kernel::write_back_mma_rC(stensor20000923_ptr, matmul_20000923_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000921OutputAtom::run(dtensor10000215_tile_ptr, stensor20000921_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20000923OutputAtom::run(dtensor10000216_tile_ptr, stensor20000923_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000217_ptr, half_t const* __restrict__ dtensor10000215_ptr, half_t const* __restrict__ dtensor10000216_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000937_ptr = (half_t*)(buf + 400);
  half_t *stensor20000936_ptr = (half_t*)(buf + 144);
  half_t *stensor20000934_ptr = (half_t*)(buf + 128);
  half_t *stensor20000935_ptr = (half_t*)(buf + 10368);
  half_t *stensor20000933_ptr = (half_t*)(buf + 6272);
  half_t *stensor20000932_ptr = (half_t*)(buf + 4224);
  half_t *stensor20000931_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000215 -> stensor 20000931
  const half_t *dtensor10000215_tile_ptr = dtensor10000215_ptr  + blockIdx.x*1*1 + blockIdx.y*2*8192;
  using DTensor10000215TileLayout = Layout<Shape<Int<1>, Int<1024>, Int<2>>, Stride<Int<1>, Int<8>, Int<8192>>>;
  using STensor20000931InputAtom = tb::InputNonChunkedSyncCopy<half_t, Layout<Shape<Int<1>, Int<1024>, Int<2>>, Stride<Int<1>, Int<1>, Int<1024>>>, DTensor10000215TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000216 -> stensor 20000932
  const half_t *dtensor10000216_tile_ptr = dtensor10000216_ptr  + blockIdx.x*1*131072 + blockIdx.y*2*512;
  using DTensor10000216TileLayout = Layout<Shape<Int<512>, Int<2>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20000932InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<512>, Int<2>, Int<1>>, Stride<Int<1>, Int<512>, Int<1024>>>, DTensor10000216TileLayout, NUM_THREADS>;
  
  STensor20000931InputAtom::run(stensor20000931_ptr, dtensor10000215_tile_ptr, thread_idx);
  STensor20000932InputAtom::run(stensor20000932_ptr, dtensor10000216_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000937 -> dtensor 10000217
  half_t *dtensor10000217_tile_ptr = dtensor10000217_ptr  + blockIdx.x*1*16384 + blockIdx.y*2*64;
  using DTensor10000217TileLayout = Layout<Shape<Int<64>, Int<2>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000937OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000217TileLayout, Layout<Shape<Int<64>, Int<2>, Int<1>>, Stride<Int<1>, Int<64>, Int<128>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000933_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 1024, NUM_THREADS>::run(stensor20000935_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<1>, Int<1024>, Int<2>>, Stride<Int<1>, Int<1>, Int<1024>>>, Layout<Shape<Int<1>, Int<1024>, Int<2>>, Stride<Int<1>, Int<1>, Int<1024>>>, NUM_THREADS>;
      Kernel::run(stensor20000933_ptr, stensor20000931_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<1>, Int<512>, Int<2>>, Stride<Int<1>, Int<1>, Int<512>>>, Layout<Shape<Int<1>, Int<512>, Int<2>>, Stride<Int<1024>, Int<1>, Int<512>>>, NUM_THREADS>;
      Kernel::run(stensor20000935_ptr, stensor20000932_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<1>, Int<1024>, Int<2>>, Stride<Int<1>, Int<1>, Int<1024>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<1>, Int<2>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000934_ptr, stensor20000933_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<1>, Int<512>, Int<2>>, Stride<Int<1>, Int<1>, Int<512>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000936_ptr, stensor20000935_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<2>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<128>, Int<1>, Int<64>>>;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000937_ptr, stensor20000936_ptr, stensor20000934_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000937OutputAtom::run(dtensor10000217_tile_ptr, stensor20000937_ptr, thread_idx);
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
    half_t *dtensor10000215 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000216 = (half_t*)((char*)buf + 4194304);
    half_t *dtensor10000212 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000213 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000214 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000215, dtensor10000216, dtensor10000212, dtensor10000213, dtensor10000214);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000217 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000215 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000216 = (half_t*)((char*)buf + 4194304);
    dim3 grid_dim(2, 128, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 12416;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 12416);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000217, dtensor10000215, dtensor10000216);
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
