#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000233_ptr, half_t* __restrict__ dtensor10000234_ptr, half_t const* __restrict__ dtensor10000230_ptr, half_t const* __restrict__ dtensor10000231_ptr, half_t const* __restrict__ dtensor10000232_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001017_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001015_ptr = (half_t*)(buf + 128);
  half_t *stensor20001013_ptr = (half_t*)(buf + 71808);
  half_t *stensor30001010_ptr = (half_t*)(buf + 55424);
  half_t *stensor20001010_ptr = (half_t*)(buf + 39040);
  half_t *stensor20001011_ptr = (half_t*)(buf + 22656);
  half_t *stensor30001011_ptr = (half_t*)(buf + 6272);
  half_t *stensor20001014_ptr = (half_t*)(buf + 2176);
  half_t *stensor20001009_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000230 -> stensor 20001009
  const half_t *dtensor10000230_tile_ptr = dtensor10000230_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000230TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001009InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000230TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000231 -> stensor 20001010
  const half_t *dtensor10000231_tile_ptr = dtensor10000231_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000231TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001010InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000231TileLayout, NUM_THREADS>;
  half_t *stensor20001010_async_copy_buf = stensor30001010_ptr;
  // Copy for G->S: dtensor 10000232 -> stensor 20001011
  const half_t *dtensor10000232_tile_ptr = dtensor10000232_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000232TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001011InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000232TileLayout, NUM_THREADS>;
  half_t *stensor20001011_async_copy_buf = stensor30001011_ptr;
  
  STensor20001009InputAtom::run(stensor20001009_ptr, dtensor10000230_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001017 -> dtensor 10000234
  half_t *dtensor10000234_tile_ptr = dtensor10000234_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*1 + blockIdx.z*16*512;
  using DTensor10000234TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20001017OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000234TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20001015 -> dtensor 10000233
  half_t *dtensor10000233_tile_ptr = dtensor10000233_ptr  + blockIdx.x*1*2048 + blockIdx.y*1*256 + blockIdx.z*16*1;
  using DTensor10000233TileLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<256>, Int<2048>>>;
  using STensor20001015OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000233TileLayout, Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20001014_ptr, thread_idx);
  
  
  using Matmul20001013LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001013LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001013LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001013LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001013LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20001013Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001013LayoutA, Matmul20001013LayoutB, Matmul20001013LayoutC, Matmul20001013LayoutAAligned, Matmul20001013LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001017LayoutA = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001017LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001017LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001017LayoutAAligned = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20001017LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001017Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20001017LayoutA, Matmul20001017LayoutB, Matmul20001017LayoutC, Matmul20001017LayoutAAligned, Matmul20001017LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001017_accum = Matmul20001017Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001011InputAtom::run(stensor20001011_async_copy_buf, dtensor10000232_tile_ptr, thread_idx);
    STensor20001010InputAtom::run(stensor20001010_async_copy_buf, dtensor10000231_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001011InputAtom::run(stensor20001011_ptr, dtensor10000232_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20001010InputAtom::run(stensor20001010_ptr, dtensor10000231_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001011_ptr, stensor20001011_async_copy_buf);
      SWAP(stensor20001010_ptr, stensor20001010_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001013Kernel::get_mma_rC(thread_idx);
      Matmul20001013Kernel::run(mma_rC, stensor20001009_ptr, stensor20001010_ptr, (char*)(buf+0), thread_idx);
      Matmul20001013Kernel::write_back_mma_rC(stensor20001013_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001014_ptr, stensor20001013_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001017Kernel::run(matmul_20001017_accum, stensor20001013_ptr, stensor20001011_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001017Kernel::write_back_mma_rC(stensor20001017_ptr, matmul_20001017_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001015_ptr, stensor20001014_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20001017OutputAtom::run(dtensor10000234_tile_ptr, stensor20001017_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001015OutputAtom::run(dtensor10000233_tile_ptr, stensor20001015_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000235_ptr, half_t const* __restrict__ dtensor10000233_ptr, half_t const* __restrict__ dtensor10000234_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001031_ptr = (half_t*)(buf + 128);
  half_t *stensor20001030_ptr = (half_t*)(buf + 1200);
  half_t *stensor20001028_ptr = (half_t*)(buf + 1184);
  half_t *stensor20001025_ptr = (half_t*)(buf + 3232);
  half_t *stensor30001026_ptr = (half_t*)(buf + 2208);
  half_t *stensor20001026_ptr = (half_t*)(buf + 1184);
  half_t *stensor20001029_ptr = (half_t*)(buf + 160);
  half_t *stensor20001027_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000233 -> stensor 20001025
  const half_t *dtensor10000233_tile_ptr = dtensor10000233_ptr  + blockIdx.x*1*2048 + blockIdx.y*4*1;
  using DTensor10000233TileLayout = Layout<Shape<Int<4>, Int<2>, Int<1>>, Stride<Int<1>, Int<256>, Int<2048>>>;
  using STensor20001025InputAtom = tb::InputNonChunkedSyncCopy<half_t, Layout<Shape<Int<4>, Int<2>, Int<1>>, Stride<Int<1>, Int<8>, Int<16>>>, DTensor10000233TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000234 -> stensor 20001026
  const half_t *dtensor10000234_tile_ptr = dtensor10000234_ptr  + blockIdx.x*1*131072 + blockIdx.y*4*512;
  using DTensor10000234TileLayout = Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20001026InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<128>, Int<512>>>, DTensor10000234TileLayout, NUM_THREADS>;
  half_t *stensor20001026_async_copy_buf = stensor30001026_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001031 -> dtensor 10000235
  half_t *dtensor10000235_tile_ptr = dtensor10000235_ptr  + blockIdx.x*1*16384 + blockIdx.y*4*64;
  using DTensor10000235TileLayout = Layout<Shape<Int<64>, Int<4>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001031OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000235TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<4>, Int<1>>, Stride<Int<1>, Int<64>, Int<256>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 16, NUM_THREADS>::run(stensor20001027_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 512, NUM_THREADS>::run(stensor20001029_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20001026InputAtom::run(stensor20001026_async_copy_buf, dtensor10000234_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001026InputAtom::run(stensor20001026_ptr, dtensor10000234_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001026_ptr, stensor20001026_async_copy_buf);
    }
    {
      // OP type: tb_input_op
      STensor20001025InputAtom::run(stensor20001025_ptr, dtensor10000233_tile_ptr + 512*for_idx, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<4>, Int<2>, Int<1>>, Stride<Int<1>, Int<8>, Int<16>>>, Layout<Shape<Int<4>, Int<2>, Int<1>>, Stride<Int<1>, Int<8>, Int<16>>>, NUM_THREADS>;
      Kernel::run(stensor20001027_ptr, stensor20001025_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, decltype(composition(Swizzle<5, 1, 6>{}, Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<128>, Int<512>>>{})), Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<128>, Int<512>>>, NUM_THREADS>;
      Kernel::run(stensor20001029_ptr, stensor20001026_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<4>, Int<2>, Int<1>>, Stride<Int<1>, Int<8>, Int<16>>>;
    using OutLayout = Layout<Shape<Int<4>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001028_ptr, stensor20001027_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = decltype(composition(Swizzle<5, 1, 6>{}, Layout<Shape<Int<4>, Int<128>, Int<1>>, Stride<Int<128>, Int<1>, Int<512>>>{}));
    using OutLayout = Layout<Shape<Int<4>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001030_ptr, stensor20001029_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<4>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using In1Layout = Layout<Shape<Int<4>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<4>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<256>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001031_ptr, stensor20001030_ptr, stensor20001028_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001031OutputAtom::run(dtensor10000235_tile_ptr, stensor20001031_ptr, thread_idx);
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
    half_t *dtensor10000233 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000234 = (half_t*)((char*)buf + 8192);
    half_t *dtensor10000230 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000231 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000232 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000233, dtensor10000234, dtensor10000230, dtensor10000231, dtensor10000232);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000235 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000233 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000234 = (half_t*)((char*)buf + 8192);
    dim3 grid_dim(2, 64, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 3264;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 3264);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000235, dtensor10000233, dtensor10000234);
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
