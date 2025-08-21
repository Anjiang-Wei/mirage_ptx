#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000311_ptr, half_t* __restrict__ dtensor10000312_ptr, half_t const* __restrict__ dtensor10000308_ptr, half_t const* __restrict__ dtensor10000309_ptr, half_t const* __restrict__ dtensor10000310_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001421_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001419_ptr = (half_t*)(buf + 128);
  half_t *stensor20001417_ptr = (half_t*)(buf + 49280);
  half_t *stensor30001414_ptr = (half_t*)(buf + 41088);
  half_t *stensor20001414_ptr = (half_t*)(buf + 32896);
  half_t *stensor20001415_ptr = (half_t*)(buf + 24704);
  half_t *stensor30001415_ptr = (half_t*)(buf + 16512);
  half_t *stensor20001418_ptr = (half_t*)(buf + 8320);
  half_t *stensor20001413_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000308 -> stensor 20001413
  const half_t *dtensor10000308_tile_ptr = dtensor10000308_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000308TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001413InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000308TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000309 -> stensor 20001414
  const half_t *dtensor10000309_tile_ptr = dtensor10000309_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*1;
  using DTensor10000309TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<4096>, Int<262144>>>;
  using STensor20001414InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000309TileLayout, NUM_THREADS>;
  half_t *stensor20001414_async_copy_buf = stensor30001414_ptr;
  // Copy for G->S: dtensor 10000310 -> stensor 20001415
  const half_t *dtensor10000310_tile_ptr = dtensor10000310_ptr  + blockIdx.x*1*262144 + blockIdx.y*256*64;
  using DTensor10000310TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<262144>>>;
  using STensor20001415InputAtom = tb::InputChunkedAsyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000310TileLayout, NUM_THREADS>;
  half_t *stensor20001415_async_copy_buf = stensor30001415_ptr;
  
  STensor20001413InputAtom::run(stensor20001413_ptr, dtensor10000308_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001421 -> dtensor 10000312
  half_t *dtensor10000312_tile_ptr = dtensor10000312_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*256 + blockIdx.z*64*1;
  using DTensor10000312TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001421OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000312TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20001419 -> dtensor 10000311
  half_t *dtensor10000311_tile_ptr = dtensor10000311_ptr  + blockIdx.x*1*4096 + blockIdx.y*1*256 + blockIdx.z*64*1;
  using DTensor10000311TileLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20001419OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000311TileLayout, Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20001418_ptr, thread_idx);
  
  
  using Matmul20001417LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001417LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001417LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001417LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001417LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001417Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001417LayoutA, Matmul20001417LayoutB, Matmul20001417LayoutC, Matmul20001417LayoutAAligned, Matmul20001417LayoutBAligned,NUM_THREADS, 1, false>;
  
  using Matmul20001421LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001421LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001421LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20001421LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001421LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20001421Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20001421LayoutA, Matmul20001421LayoutB, Matmul20001421LayoutC, Matmul20001421LayoutAAligned, Matmul20001421LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20001421_accum = Matmul20001421Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20001415InputAtom::run(stensor20001415_async_copy_buf, dtensor10000310_tile_ptr, thread_idx);
    STensor20001414InputAtom::run(stensor20001414_async_copy_buf, dtensor10000309_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 4) {
        STensor20001415InputAtom::run(stensor20001415_ptr, dtensor10000310_tile_ptr + 4096*(for_idx+1), thread_idx);
        STensor20001414InputAtom::run(stensor20001414_ptr, dtensor10000309_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20001415_ptr, stensor20001415_async_copy_buf);
      SWAP(stensor20001414_ptr, stensor20001414_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20001417Kernel::get_mma_rC(thread_idx);
      Matmul20001417Kernel::run(mma_rC, stensor20001413_ptr, stensor20001414_ptr, (char*)(buf+0), thread_idx);
      Matmul20001417Kernel::write_back_mma_rC(stensor20001417_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20001418_ptr, stensor20001417_ptr, thread_idx);
    }
    {
      // OP type: tb_matmul_op
      Matmul20001421Kernel::run(matmul_20001421_accum, stensor20001417_ptr, stensor20001415_ptr, (char*)(buf+0), thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20001421Kernel::write_back_mma_rC(stensor20001421_ptr, matmul_20001421_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001419_ptr, stensor20001418_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20001421OutputAtom::run(dtensor10000312_tile_ptr, stensor20001421_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001419OutputAtom::run(dtensor10000311_tile_ptr, stensor20001419_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000313_ptr, half_t const* __restrict__ dtensor10000311_ptr, half_t const* __restrict__ dtensor10000312_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20001435_ptr = (half_t*)(buf + 2208);
  half_t *stensor20001434_ptr = (half_t*)(buf + 160);
  half_t *stensor20001432_ptr = (half_t*)(buf + 128);
  half_t *stensor20001433_ptr = (half_t*)(buf + 33920);
  half_t *stensor20001431_ptr = (half_t*)(buf + 33408);
  half_t *stensor20001430_ptr = (half_t*)(buf + 640);
  half_t *stensor20001429_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000311 -> stensor 20001429
  const half_t *dtensor10000311_tile_ptr = dtensor10000311_ptr  + blockIdx.x*1*4096 + blockIdx.y*16*1;
  using DTensor10000311TileLayout = Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20001429InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<16>, Int<256>>>, DTensor10000311TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000312 -> stensor 20001430
  const half_t *dtensor10000312_tile_ptr = dtensor10000312_ptr  + blockIdx.x*1*262144 + blockIdx.y*16*1;
  using DTensor10000312TileLayout = Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20001430InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, DTensor10000312TileLayout, NUM_THREADS>;
  
  STensor20001429InputAtom::run(stensor20001429_ptr, dtensor10000311_tile_ptr, thread_idx);
  STensor20001430InputAtom::run(stensor20001430_ptr, dtensor10000312_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20001435 -> dtensor 10000313
  half_t *dtensor10000313_tile_ptr = dtensor10000313_ptr  + blockIdx.x*1*16384 + blockIdx.y*16*64;
  using DTensor10000313TileLayout = Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20001435OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000313TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>, Int<1>>, Stride<Int<1>, Int<64>, Int<1024>>>{})), NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 256, NUM_THREADS>::run(stensor20001431_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 16384, NUM_THREADS>::run(stensor20001433_ptr, thread_idx);
  
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<16>, Int<256>>>, Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<16>, Int<256>>>, NUM_THREADS>;
      Kernel::run(stensor20001431_ptr, stensor20001429_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, NUM_THREADS>;
      Kernel::run(stensor20001433_ptr, stensor20001430_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<16>, Int<256>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001432_ptr, stensor20001431_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001434_ptr, stensor20001433_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using In1Layout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<1024>>>{}));
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20001435_ptr, stensor20001434_ptr, stensor20001432_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20001435OutputAtom::run(dtensor10000313_tile_ptr, stensor20001435_ptr, thread_idx);
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
    half_t *dtensor10000311 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000312 = (half_t*)((char*)buf + 16384);
    half_t *dtensor10000308 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000309 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000310 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 57472;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 57472);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000311, dtensor10000312, dtensor10000308, dtensor10000309, dtensor10000310);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000313 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000311 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000312 = (half_t*)((char*)buf + 16384);
    dim3 grid_dim(2, 16, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 66688;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 66688);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000313, dtensor10000311, dtensor10000312);
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
