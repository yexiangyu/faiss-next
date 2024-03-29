use crate::error::Result;
use crate::macros::rc;

pub fn get_num_gpus() -> Result<i32> {
    let mut num = 0;
    rc!({ faiss_next_sys::faiss_get_num_gpus(&mut num) })?;
    Ok(num)
}

pub fn gpu_profiler_start() -> Result<()> {
    rc!({ faiss_next_sys::faiss_gpu_profiler_start() })?;
    Ok(())
}

pub fn gpu_profiler_stop() -> Result<()> {
    rc!({ faiss_next_sys::faiss_gpu_profiler_stop() })?;
    Ok(())
}

pub fn gpu_sync_all_devices() -> Result<()> {
    rc!({ faiss_next_sys::faiss_gpu_sync_all_devices() })?;
    Ok(())
}
