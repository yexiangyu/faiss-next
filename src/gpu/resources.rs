use std::ffi::c_void;
use std::ptr::null_mut;

use crate::error::Result;
use crate::macros::rc;
use faiss_next_sys as sys;

pub trait GpuResourcesTrait {
    fn ptr(&self) -> *mut sys::FaissGpuResources;

    fn initialize_for_device(&self, device: i32) -> Result<()> {
        rc!({ sys::faiss_GpuResources_initializeForDevice(self.ptr(), device) })
    }

    fn get_blas_handle(&self, device: i32) -> Result<sys::cublasHandle_t> {
        let mut handle = null_mut();
        rc!({ sys::faiss_GpuResources_getBlasHandle(self.ptr(), device, &mut handle) })?;
        Ok(handle)
    }

    fn get_default_stream(&self, device: i32) -> Result<sys::cudaStream_t> {
        let mut stream = null_mut();
        rc!({ sys::faiss_GpuResources_getDefaultStream(self.ptr(), device, &mut stream) })?;
        Ok(stream)
    }

    fn get_pinned_memory(&self) -> Result<(*mut c_void, usize)> {
        let mut ptr = null_mut();
        let mut size = 0usize;
        rc!({ sys::faiss_GpuResources_getPinnedMemory(self.ptr(), &mut ptr, &mut size) })?;
        Ok((ptr, size))
    }

    fn get_async_copy_stream(&self, device: i32) -> Result<sys::cudaStream_t> {
        let mut stream = null_mut();
        rc!({ sys::faiss_GpuResources_getAsyncCopyStream(self.ptr(), device, &mut stream) })?;
        Ok(stream)
    }

    fn get_blas_handle_current_device(&self) -> Result<sys::cublasHandle_t> {
        let mut handle = null_mut();
        rc!({ sys::faiss_GpuResources_getBlasHandleCurrentDevice(self.ptr(), &mut handle) })?;
        Ok(handle)
    }

    fn sync_default_stream(&self, device: i32) -> Result<()> {
        rc!({ sys::faiss_GpuResources_syncDefaultStream(self.ptr(), device) })
    }

    fn get_default_stream_current_device(&self) -> Result<sys::cudaStream_t> {
        let mut stream = null_mut();
        rc!({ sys::faiss_GpuResources_getDefaultStreamCurrentDevice(self.ptr(), &mut stream) })?;
        Ok(stream)
    }

    fn sync_default_stream_current_device(&self) -> Result<()> {
        rc!({ sys::faiss_GpuResources_syncDefaultStreamCurrentDevice(self.ptr()) })
    }

    fn get_async_copy_stream_current_device(&self) -> Result<sys::cudaStream_t> {
        let mut stream = null_mut();
        rc!({ sys::faiss_GpuResources_getAsyncCopyStreamCurrentDevice(self.ptr(), &mut stream) })?;
        Ok(stream)
    }
}

pub struct GpuResources {
    inner: *mut sys::FaissGpuResources,
}

impl GpuResourcesTrait for GpuResources {
    fn ptr(&self) -> *mut sys::FaissGpuResources {
        self.inner
    }
}

pub trait GpuResourcesProviderTrait {
    fn ptr(&self) -> *mut sys::FaissGpuResourcesProvider;

    fn get_resources(&self) -> Result<GpuResources> {
        let mut inner = null_mut();
        rc!({ sys::faiss_GpuResourcesProvider_getResources(self.ptr(), &mut inner) })?;
        Ok(GpuResources { inner })
    }
}
