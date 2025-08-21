#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000221_ptr, half_t* __restrict__ dtensor10000222_ptr, half_t const* __restrict__ dtensor10000218_ptr, half_t const* __restrict__ dtensor10000219_ptr, half_t const* __restrict__ dtensor10000220_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000953_ptr = (half_t*)(buf + 128);
  half_t *stensor20000950_ptr = (half_t*)(buf + 71808);
  half_t *stensor30000948_ptr = (half_t*)(buf + 55424);
  half_t *stensor20000948_ptr = (half_t*)(buf + 39040);
  half_t *stensor30000947_ptr = (half_t*)(buf + 22656);
  half_t *stensor20000947_ptr = (half_t*)(buf + 6272);
  half_t *stensor20000951_ptr = (half_t*)(buf + 2176);
  half_t *stensor20000946_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000218 -> stensor 20000946
  const half_t *dtensor10000218_tile_ptr = dtensor10000218_ptr  + blockIdx.x*1*16384 + blockIdx.z*16*64;
  using DTensor10000218TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000946InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), DTensor10000218TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000219 -> stensor 20000947
  const half_t *dtensor10000219_tile_ptr = dtensor10000219_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*1;
  using DTensor10000219TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20000947InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000219TileLayout, NUM_THREADS>;
  half_t *stensor20000947_async_copy_buf = stensor30000947_ptr;
  // Copy for G->S: dtensor 10000220 -> stensor 20000948
  const half_t *dtensor10000220_tile_ptr = dtensor10000220_ptr  + blockIdx.x*1*262144 + blockIdx.y*512*64;
  using DTensor10000220TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20000948InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000220TileLayout, NUM_THREADS>;
  half_t *stensor20000948_async_copy_buf = stensor30000948_ptr;
  
  STensor20000946InputAtom::run(stensor20000946_ptr, dtensor10000218_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000951 -> dtensor 10000221
  half_t *dtensor10000221_tile_ptr = dtensor10000221_ptr  + blockIdx.x*1*1 + blockIdx.y*128*8 + blockIdx.z*16*8192;
  using DTensor10000221TileLayout = Layout<Shape<Int<1>, Int<128>, Int<16>>, Stride<Int<1>, Int<8>, Int<8192>>>;
  using STensor20000951OutputAtom = tb::OutputNonChunkedSyncCopy<half_t, DTensor10000221TileLayout, Layout<Shape<Int<1>, Int<128>, Int<16>>, Stride<Int<1>, Int<1>, Int<128>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20000953 -> dtensor 10000222
  half_t *dtensor10000222_tile_ptr = dtensor10000222_ptr  + blockIdx.x*1*131072 + blockIdx.y*64*1 + blockIdx.z*16*512;
  using DTensor10000222TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20000953OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000222TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 2048, NUM_THREADS>::run(stensor20000951_ptr, thread_idx);
  
  
  using Matmul20000950LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000950LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000950LayoutC = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000950LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000950LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000950Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000950LayoutA, Matmul20000950LayoutB, Matmul20000950LayoutC, Matmul20000950LayoutAAligned, Matmul20000950LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20000953LayoutA = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000953LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000953LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000953LayoutAAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
  using Matmul20000953LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000953Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000953LayoutA, Matmul20000953LayoutB, Matmul20000953LayoutC, Matmul20000953LayoutAAligned, Matmul20000953LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000953_accum = Matmul20000953Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20000948InputAtom::run(stensor20000948_async_copy_buf, dtensor10000220_tile_ptr, thread_idx);
    STensor20000947InputAtom::run(stensor20000947_async_copy_buf, dtensor10000219_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20000948InputAtom::run(stensor20000948_ptr, dtensor10000220_tile_ptr + 8192*(for_idx+1), thread_idx);
        STensor20000947InputAtom::run(stensor20000947_ptr, dtensor10000219_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000948_ptr, stensor20000948_async_copy_buf);
      SWAP(stensor20000947_ptr, stensor20000947_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20000950Kernel::get_mma_rC(thread_idx);
      Matmul20000950Kernel::run(mma_rC, stensor20000946_ptr, stensor20000947_ptr, (char*)(buf+0), thread_idx);
      Matmul20000950Kernel::write_back_mma_rC(stensor20000950_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<128>, Int<16>, Int<1>>, Stride<Int<1>, Int<128>, Int<1>>>, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>, Int<1>>, Stride<Int<1>, Int<128>, Int<2048>>>{})), NUM_THREADS>;
      Kernel::run(stensor20000951_ptr, stensor20000950_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20000953Kernel::run(matmul_20000953_accum, stensor20000950_ptr, stensor20000948_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000953Kernel::write_back_mma_rC(stensor20000953_ptr, matmul_20000953_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000951OutputAtom::run(dtensor10000221_tile_ptr, stensor20000951_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20000953OutputAtom::run(dtensor10000222_tile_ptr, stensor20000953_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000223_ptr, half_t const* __restrict__ dtensor10000221_ptr, half_t const* __restrict__ dtensor10000222_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000967_ptr = (half_t*)(buf + 128);
  half_t *stensor20000966_ptr = (half_t*)(buf + 1680);
  half_t *stensor20000964_ptr = (half_t*)(buf + 1664);
  half_t *stensor20000961_ptr = (half_t*)(buf + 2688);
  half_t *stensor30000962_ptr = (half_t*)(buf + 2176);
  half_t *stensor20000962_ptr = (half_t*)(buf + 1664);
  half_t *stensor20000965_ptr = (half_t*)(buf + 1152);
  half_t *stensor20000963_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000221 -> stensor 20000961
  const half_t *dtensor10000221_tile_ptr = dtensor10000221_ptr  + blockIdx.x*1*1 + blockIdx.y*2*8192;
  using DTensor10000221TileLayout = Layout<Shape<Int<1>, Int<256>, Int<2>>, Stride<Int<1>, Int<8>, Int<8192>>>;
  using STensor20000961InputAtom = tb::InputNonChunkedSyncCopy<half_t, Layout<Shape<Int<1>, Int<256>, Int<2>>, Stride<Int<1>, Int<1>, Int<256>>>, DTensor10000221TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000222 -> stensor 20000962
  const half_t *dtensor10000222_tile_ptr = dtensor10000222_ptr  + blockIdx.x*1*131072 + blockIdx.y*2*512;
  using DTensor10000222TileLayout = Layout<Shape<Int<128>, Int<2>, Int<1>>, Stride<Int<1>, Int<512>, Int<131072>>>;
  using STensor20000962InputAtom = tb::InputChunkedAsyncCopy<half_t, Layout<Shape<Int<128>, Int<2>, Int<1>>, Stride<Int<1>, Int<128>, Int<256>>>, DTensor10000222TileLayout, NUM_THREADS>;
  half_t *stensor20000962_async_copy_buf = stensor30000962_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000967 -> dtensor 10000223
  half_t *dtensor10000223_tile_ptr = dtensor10000223_ptr  + blockIdx.x*1*16384 + blockIdx.y*2*64;
  using DTensor10000223TileLayout = Layout<Shape<Int<64>, Int<2>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000967OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000223TileLayout, Layout<Shape<Int<64>, Int<2>, Int<1>>, Stride<Int<1>, Int<64>, Int<128>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 512, NUM_THREADS>::run(stensor20000963_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 256, NUM_THREADS>::run(stensor20000965_ptr, thread_idx);
  
  
  __syncthreads();
  
  {
    STensor20000962InputAtom::run(stensor20000962_async_copy_buf, dtensor10000222_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20000962InputAtom::run(stensor20000962_ptr, dtensor10000222_tile_ptr + 128*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000962_ptr, stensor20000962_async_copy_buf);
    }
    {
      // OP type: tb_input_op
      STensor20000961InputAtom::run(stensor20000961_ptr, dtensor10000221_tile_ptr + 2048*for_idx, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<1>, Int<256>, Int<2>>, Stride<Int<1>, Int<1>, Int<256>>>, Layout<Shape<Int<1>, Int<256>, Int<2>>, Stride<Int<1>, Int<1>, Int<256>>>, NUM_THREADS>;
      Kernel::run(stensor20000963_ptr, stensor20000961_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<128>, Int<2>, Int<1>>, Stride<Int<1>, Int<128>, Int<1>>>, Layout<Shape<Int<128>, Int<2>, Int<1>>, Stride<Int<1>, Int<128>, Int<256>>>, NUM_THREADS>;
      Kernel::run(stensor20000965_ptr, stensor20000962_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<1>, Int<256>, Int<2>>, Stride<Int<1>, Int<1>, Int<256>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<1>, Int<2>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000964_ptr, stensor20000963_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<1>, Int<128>, Int<2>>, Stride<Int<1>, Int<1>, Int<128>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000966_ptr, stensor20000965_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<1>, Int<1>, Int<64>>>;
    using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<2>>, Stride<Int<1>, Int<1>, Int<1>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>, Int<2>>, Stride<Int<128>, Int<1>, Int<64>>>;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000967_ptr, stensor20000966_ptr, stensor20000964_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000967OutputAtom::run(dtensor10000223_tile_ptr, stensor20000967_ptr, thread_idx);
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
    half_t *dtensor10000221 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000222 = (half_t*)((char*)buf + 4194304);
    half_t *dtensor10000218 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000219 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000220 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 8, 16);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 75904;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 75904);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000221, dtensor10000222, dtensor10000218, dtensor10000219, dtensor10000220);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000223 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000221 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000222 = (half_t*)((char*)buf + 4194304);
    dim3 grid_dim(2, 128, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 3712;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 3712);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000223, dtensor10000221, dtensor10000222);
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
