#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000197_ptr, half_t* __restrict__ dtensor10000198_ptr, half_t const* __restrict__ dtensor10000194_ptr, half_t const* __restrict__ dtensor10000195_ptr, half_t const* __restrict__ dtensor10000196_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000833_ptr = (half_t*)(buf + 128);
  half_t *stensor20000830_ptr = (half_t*)(buf + 71808);
  half_t *stensor30000828_ptr = (half_t*)(buf + 55424);
  half_t *stensor20000828_ptr = (half_t*)(buf + 39040);
  half_t *stensor30000827_ptr = (half_t*)(buf + 22656);
  half_t *stensor20000827_ptr = (half_t*)(buf + 6272);
  half_t *stensor20000831_ptr = (half_t*)(buf + 2176);
  half_t *stensor20000826_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000194 -> stensor 20000826
  const half_t *dtensor10000194_tile_ptr = dtensor10000194_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000194TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000826InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000194TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000195 -> stensor 20000827
  const half_t *dtensor10000195_tile_ptr = dtensor10000195_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000195TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20000827InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000195TileLayout, NUM_THREADS>;
  half_t *stensor20000827_async_copy_buf = stensor30000827_ptr;
  // Copy for G->S: dtensor 10000196 -> stensor 20000828
  const half_t *dtensor10000196_tile_ptr = dtensor10000196_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000196TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20000828InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000196TileLayout, NUM_THREADS>;
  half_t *stensor20000828_async_copy_buf = stensor30000828_ptr;
  
  STensor20000826InputAtom::run(stensor20000826_ptr, dtensor10000194_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000831 -> dtensor 10000197
  half_t *dtensor10000197_tile_ptr = dtensor10000197_ptr  + blockIdx.x*1*1 + blockIdx.y*128*8 + blockIdx.z*16*8192;
  using DTensor10000197TileLayout = Layout<Shape<Int<1>, Int<128>, Int<16>>, Stride<Int<1>, Int<8>, Int<8192>>>;
  using STensor20000831OutputAtom = tb::OutputNonChunkedSyncCopy<half_t, DTensor10000197TileLayout, Layout<Shape<Int<1>, Int<128>, Int<16>>, Stride<Int<1>, Int<1>, Int<128>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20000833 -> dtensor 10000198
  half_t *dtensor10000198_tile_ptr = dtensor10000198_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*1 + blockIdx.z*16*512;
  using DTensor10000198TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20000833OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000198TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000831_ptr, thread_idx);
  
  
  using Matmul20000830LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000830LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000830LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20000830LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000830LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000830Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000830LayoutA, Matmul20000830LayoutB, Matmul20000830LayoutC, Matmul20000830LayoutAAligned, Matmul20000830LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20000833LayoutA = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20000833LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000833LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000833LayoutAAligned = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20000833LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000833Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000833LayoutA, Matmul20000833LayoutB, Matmul20000833LayoutC, Matmul20000833LayoutAAligned, Matmul20000833LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000833_accum = Matmul20000833Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20000828InputAtom::run(stensor20000828_async_copy_buf, dtensor10000196_tile_ptr, thread_idx);
    STensor20000827InputAtom::run(stensor20000827_async_copy_buf, dtensor10000195_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20000828InputAtom::run(stensor20000828_ptr, dtensor10000196_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20000827InputAtom::run(stensor20000827_ptr, dtensor10000195_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000828_ptr, stensor20000828_async_copy_buf);
      SWAP(stensor20000827_ptr, stensor20000827_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20000830Kernel::get_mma_rC(thread_idx);
      Matmul20000830Kernel::run(mma_rC, stensor20000826_ptr, stensor20000827_ptr, (char*)(buf+0), thread_idx);
      Matmul20000830Kernel::write_back_mma_rC(stensor20000830_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<128>, Int<16>, Int<1>>, Stride<Int<1>, Int<128>, Int<1>>>, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>, Int<1>>, Stride<Int<16>, Int<1>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20000831_ptr, stensor20000830_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20000833Kernel::run(matmul_20000833_accum, stensor20000830_ptr, stensor20000828_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000833Kernel::write_back_mma_rC(stensor20000833_ptr, matmul_20000833_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000831OutputAtom::run(dtensor10000197_tile_ptr, stensor20000831_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20000833OutputAtom::run(dtensor10000198_tile_ptr, stensor20000833_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000199_ptr, half_t const* __restrict__ dtensor10000197_ptr, half_t const* __restrict__ dtensor10000198_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000847_ptr = (half_t*)(buf + 128);
  half_t *stensor20000846_ptr = (half_t*)(buf + 3216);
  half_t *stensor20000844_ptr = (half_t*)(buf + 3200);
  half_t *stensor20000841_ptr = (half_t*)(buf + 5248);
  half_t *stensor30000842_ptr = (half_t*)(buf + 4224);
  half_t *stensor20000842_ptr = (half_t*)(buf + 3200);
  half_t *stensor20000845_ptr = (half_t*)(buf + 2176);
  half_t *stensor20000843_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000197 -> stensor 20000841
  const half_t *dtensor10000197_tile_ptr = dtensor10000197_ptr  + blockIdx.x*1*1 + blockIdx.y*4*8192;
  using DTensor10000197TileLayout = Layout<Shape<Int<1>, Int<256>, Int<4>>, Stride<Int<1>, Int<8>, Int<8192>>>;
  using STensor20000841InputAtom = tb::InputNonChunkedSyncCopy<half_t, Layout<Shape<Int<1>, Int<256>, Int<4>>, Stride<Int<1>, Int<1>, Int<256>>>, DTensor10000197TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000198 -> stensor 20000842
  const half_t *dtensor10000198_tile_ptr = dtensor10000198_ptr  + blockIdx.x*1*131072 + blockIdx.y*4*512;
  using DTensor10000198TileLayout = Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20000842InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<128>, Int<512>>>, DTensor10000198TileLayout, NUM_THREADS>;
  half_t *stensor20000842_async_copy_buf = stensor30000842_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000847 -> dtensor 10000199
  half_t *dtensor10000199_tile_ptr = dtensor10000199_ptr  + blockIdx.x*1*16384 + blockIdx.y*4*64;
  using DTensor10000199TileLayout = Layout<Shape<Int<64>, Int<4>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000847OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000199TileLayout, Layout<Shape<Int<64>, Int<4>, Int<1>>, Stride<Int<1>, Int<64>, Int<256>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 1024, NUM_THREADS>::run(stensor20000843_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 512, NUM_THREADS>::run(stensor20000845_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20000842InputAtom::run(stensor20000842_async_copy_buf, dtensor10000198_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20000842InputAtom::run(stensor20000842_ptr, dtensor10000198_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000842_ptr, stensor20000842_async_copy_buf);
    }
    {
      // OP type: tb_input_op
      STensor20000841InputAtom::run(stensor20000841_ptr, dtensor10000197_tile_ptr + 2048*for_idx, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<1>, Int<256>, Int<4>>, Stride<Int<1>, Int<1>, Int<256>>>, Layout<Shape<Int<1>, Int<256>, Int<4>>, Stride<Int<1>, Int<1>, Int<256>>>, NUM_THREADS>;
      Kernel::run(stensor20000843_ptr, stensor20000841_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<1>, Int<128>, Int<4>>, Stride<Int<1>, Int<1>, Int<128>>>, Layout<Shape<Int<1>, Int<128>, Int<4>>, Stride<Int<512>, Int<1>, Int<128>>>, NUM_THREADS>;
      Kernel::run(stensor20000845_ptr, stensor20000842_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<1>, Int<256>, Int<4>>, Stride<Int<1>, Int<1>, Int<256>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<1>, Int<4>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000844_ptr, stensor20000843_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<1>, Int<128>, Int<4>>, Stride<Int<1>, Int<1>, Int<128>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<4>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000846_ptr, stensor20000845_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<1>, Int<64>, Int<4>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<4>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<4>>, Stride<Int<256>, Int<1>, Int<64>>>;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000847_ptr, stensor20000846_ptr, stensor20000844_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000847OutputAtom::run(dtensor10000199_tile_ptr, stensor20000847_ptr, thread_idx);
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
    half_t *dtensor10000197 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000198 = (half_t*)((char*)buf + 4194304);
    half_t *dtensor10000194 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000195 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000196 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000197, dtensor10000198, dtensor10000194, dtensor10000195, dtensor10000196);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000199 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000197 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000198 = (half_t*)((char*)buf + 4194304);
    dim3 grid_dim(2, 64, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 7296;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 7296);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000199, dtensor10000197, dtensor10000198);
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
