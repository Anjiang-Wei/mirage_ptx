#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000209_ptr, half_t* __restrict__ dtensor10000210_ptr, half_t const* __restrict__ dtensor10000206_ptr, half_t const* __restrict__ dtensor10000207_ptr, half_t const* __restrict__ dtensor10000208_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000893_ptr = (half_t*)(buf + 128);
  half_t *stensor20000890_ptr = (half_t*)(buf + 71808);
  half_t *stensor30000888_ptr = (half_t*)(buf + 55424);
  half_t *stensor20000888_ptr = (half_t*)(buf + 39040);
  half_t *stensor30000887_ptr = (half_t*)(buf + 22656);
  half_t *stensor20000887_ptr = (half_t*)(buf + 6272);
  half_t *stensor20000891_ptr = (half_t*)(buf + 2176);
  half_t *stensor20000886_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000206 -> stensor 20000886
  const half_t *dtensor10000206_tile_ptr = dtensor10000206_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000206TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000886InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000206TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000207 -> stensor 20000887
  const half_t *dtensor10000207_tile_ptr = dtensor10000207_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000207TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20000887InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000207TileLayout, NUM_THREADS>;
  half_t *stensor20000887_async_copy_buf = stensor30000887_ptr;
  // Copy for G->S: dtensor 10000208 -> stensor 20000888
  const half_t *dtensor10000208_tile_ptr = dtensor10000208_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000208TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20000888InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000208TileLayout, NUM_THREADS>;
  half_t *stensor20000888_async_copy_buf = stensor30000888_ptr;
  
  STensor20000886InputAtom::run(stensor20000886_ptr, dtensor10000206_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000891 -> dtensor 10000209
  half_t *dtensor10000209_tile_ptr = dtensor10000209_ptr  + blockIdx.x*1*262144 + blockIdx.y*128*256 + blockIdx.z*16*1;
  using DTensor10000209TileLayout = Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20000891OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000209TileLayout, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20000893 -> dtensor 10000210
  half_t *dtensor10000210_tile_ptr = dtensor10000210_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*256 + blockIdx.z*16*1;
  using DTensor10000210TileLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20000893OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000210TileLayout, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000891_ptr, thread_idx);
  
  
  using Matmul20000890LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000890LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000890LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20000890LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000890LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000890Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000890LayoutA, Matmul20000890LayoutB, Matmul20000890LayoutC, Matmul20000890LayoutAAligned, Matmul20000890LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20000893LayoutA = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20000893LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000893LayoutC = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20000893LayoutAAligned = decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<16>, Int<1>>>{}));
  using Matmul20000893LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000893Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000893LayoutA, Matmul20000893LayoutB, Matmul20000893LayoutC, Matmul20000893LayoutAAligned, Matmul20000893LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000893_accum = Matmul20000893Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20000888InputAtom::run(stensor20000888_async_copy_buf, dtensor10000208_tile_ptr, thread_idx);
    STensor20000887InputAtom::run(stensor20000887_async_copy_buf, dtensor10000207_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20000888InputAtom::run(stensor20000888_ptr, dtensor10000208_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20000887InputAtom::run(stensor20000887_ptr, dtensor10000207_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000888_ptr, stensor20000888_async_copy_buf);
      SWAP(stensor20000887_ptr, stensor20000887_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20000890Kernel::get_mma_rC(thread_idx);
      Matmul20000890Kernel::run(mma_rC, stensor20000886_ptr, stensor20000887_ptr, (char*)(buf+0), thread_idx);
      Matmul20000890Kernel::write_back_mma_rC(stensor20000890_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>, decltype(composition(Swizzle<1, 3, 3>{}, Layout<Shape<Int<16>, Int<128>, Int<1>>, Stride<Int<1>, Int<16>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20000891_ptr, stensor20000890_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20000893Kernel::run(matmul_20000893_accum, stensor20000890_ptr, stensor20000888_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000893Kernel::write_back_mma_rC(stensor20000893_ptr, matmul_20000893_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000891OutputAtom::run(dtensor10000209_tile_ptr, stensor20000891_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20000893OutputAtom::run(dtensor10000210_tile_ptr, stensor20000893_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000211_ptr, half_t const* __restrict__ dtensor10000209_ptr, half_t const* __restrict__ dtensor10000210_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000907_ptr = (half_t*)(buf + 128);
  half_t *stensor20000906_ptr = (half_t*)(buf + 6288);
  half_t *stensor20000904_ptr = (half_t*)(buf + 6272);
  half_t *stensor30000901_ptr = (half_t*)(buf + 14464);
  half_t *stensor20000901_ptr = (half_t*)(buf + 10368);
  half_t *stensor20000902_ptr = (half_t*)(buf + 8320);
  half_t *stensor30000902_ptr = (half_t*)(buf + 6272);
  half_t *stensor20000905_ptr = (half_t*)(buf + 4224);
  half_t *stensor20000903_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000209 -> stensor 20000901
  const half_t *dtensor10000209_tile_ptr = dtensor10000209_ptr  + blockIdx.x*1*262144 + blockIdx.y*8*1;
  using DTensor10000209TileLayout = Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20000901InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>, DTensor10000209TileLayout, NUM_THREADS>;
  half_t *stensor20000901_async_copy_buf = stensor30000901_ptr;
  // Copy for G->S: dtensor 10000210 -> stensor 20000902
  const half_t *dtensor10000210_tile_ptr = dtensor10000210_ptr  + blockIdx.x*1*131072 + blockIdx.y*8*1;
  using DTensor10000210TileLayout = Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<256>, Int<131072>>>;
  using STensor20000902InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>, DTensor10000210TileLayout, NUM_THREADS>;
  half_t *stensor20000902_async_copy_buf = stensor30000902_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000907 -> dtensor 10000211
  half_t *dtensor10000211_tile_ptr = dtensor10000211_ptr  + blockIdx.x*1*16384 + blockIdx.y*8*64;
  using DTensor10000211TileLayout = Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000907OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000211TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<8>, Int<1>>, Stride<Int<1>, Int<64>, Int<512>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000903_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 1024, NUM_THREADS>::run(stensor20000905_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20000902InputAtom::run(stensor20000902_async_copy_buf, dtensor10000210_tile_ptr, thread_idx);
    STensor20000901InputAtom::run(stensor20000901_async_copy_buf, dtensor10000209_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20000902InputAtom::run(stensor20000902_ptr, dtensor10000210_tile_ptr + 32768*(for_idx+1), thread_idx);
        STensor20000901InputAtom::run(stensor20000901_ptr, dtensor10000209_tile_ptr + 65536*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000902_ptr, stensor20000902_async_copy_buf);
      SWAP(stensor20000901_ptr, stensor20000901_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>, Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>, NUM_THREADS>;
      Kernel::run(stensor20000903_ptr, stensor20000901_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>, Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>, NUM_THREADS>;
      Kernel::run(stensor20000905_ptr, stensor20000902_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<8>, Int<256>, Int<1>>, Stride<Int<1>, Int<8>, Int<2048>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000904_ptr, stensor20000903_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<8>, Int<128>, Int<1>>, Stride<Int<1>, Int<8>, Int<1024>>>;
    using OutLayout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000906_ptr, stensor20000905_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<1>, Int<8>, Int<512>>>;
    using In1Layout = Layout<Shape<Int<8>, Int<1>, Int<1>>, Stride<Int<1>, Int<8>, Int<8>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<8>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<512>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000907_ptr, stensor20000906_ptr, stensor20000904_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000907OutputAtom::run(dtensor10000211_tile_ptr, stensor20000907_ptr, thread_idx);
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
    half_t *dtensor10000209 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000210 = (half_t*)((char*)buf + 1048576);
    half_t *dtensor10000206 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000207 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000208 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000209, dtensor10000210, dtensor10000206, dtensor10000207, dtensor10000208);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000211 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000209 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000210 = (half_t*)((char*)buf + 1048576);
    dim3 grid_dim(2, 32, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 18560;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 18560);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000211, dtensor10000209, dtensor10000210);
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
