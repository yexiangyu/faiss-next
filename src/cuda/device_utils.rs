use crate::error::*;
use faiss_next_sys as ffi;

/// Returns the number of available GPU devices
pub fn faiss_get_num_gpus() -> i32 {
    let mut output: i32 = 0;
    unsafe { ffi::faiss_get_num_gpus(&mut output) };
    output
}

/// Starts the CUDA profiler (exposed via SWIG)
pub fn faiss_gpu_profiler_start() -> Result<()> {
    faiss_rc(unsafe { ffi::faiss_gpu_profiler_start() })
}

/// Stops the CUDA profiler (exposed via SWIG)
pub fn faiss_gpu_profiler_stop() -> Result<()> {
    faiss_rc(unsafe { ffi::faiss_gpu_profiler_stop() })
}

/// Synchronizes the CPU against all devices (equivalent to cudaDeviceSynchronize for each device)
pub fn faiss_gpu_sync_all_devices() -> Result<()> {
    faiss_rc(unsafe { ffi::faiss_gpu_sync_all_devices() })
}
