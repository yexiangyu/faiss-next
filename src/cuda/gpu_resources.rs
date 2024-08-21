use std::{ffi::c_void, ptr::null_mut};

use crate::{error::*, macros::*};
use faiss_next_sys as ffi;

// pub use ffi::cublasContext;

#[allow(non_camel_case_types)]
pub struct cublasContext {
    pub inner: *mut ffi::cublasContext,
}
#[allow(non_camel_case_types)]
pub struct cudaStream {
    pub inner: *mut ffi::CUstream_st,
}

#[derive(Debug)]
pub struct FaissGpuResources {
    pub inner: *mut ffi::FaissGpuResources,
}
impl_faiss_drop!(FaissGpuResources, faiss_GpuResources_free);
impl FaissGpuResources {
    pub fn initialize_for_device(&mut self, device: i32) -> Result<()> {
        faiss_rc(unsafe { ffi::faiss_GpuResources_initializeForDevice(self.inner, device) })
    }
    pub fn get_blas_handle(&self, device: i32) -> Result<cublasContext> {
        let mut handle = null_mut();
        faiss_rc(unsafe {
            ffi::faiss_GpuResources_getBlasHandle(self.inner, device, &mut handle)
        })?;
        Ok(cublasContext { inner: handle })
    }
    pub fn get_default_stream(&self, device: i32) -> Result<cudaStream> {
        let mut stream = null_mut();
        faiss_rc(unsafe {
            ffi::faiss_GpuResources_getDefaultStream(self.inner, device, &mut stream)
        })?;
        Ok(cudaStream { inner: stream })
    }
    pub fn get_pinned_memory(&self) -> Result<(usize, *mut c_void)> {
        let mut size = 0;
        let mut ptr = null_mut();
        faiss_rc(unsafe {
            ffi::faiss_GpuResources_getPinnedMemory(self.inner, &mut ptr, &mut size)
        })?;
        Ok((size, ptr))
    }

    pub fn get_async_copy_stream(&self, device: i32) -> Result<cudaStream> {
        let mut stream = null_mut();
        faiss_rc(unsafe {
            ffi::faiss_GpuResources_getAsyncCopyStream(self.inner, device, &mut stream)
        })?;
        Ok(cudaStream { inner: stream })
    }

    pub fn get_blas_handle_current_device(&self) -> Result<cublasContext> {
        let mut handle = null_mut();
        faiss_rc(unsafe {
            ffi::faiss_GpuResources_getBlasHandleCurrentDevice(self.inner, &mut handle)
        })?;
        Ok(cublasContext { inner: handle })
    }

    pub fn get_default_stream_current_device(&self) -> Result<cudaStream> {
        let mut stream = null_mut();
        faiss_rc(unsafe {
            ffi::faiss_GpuResources_getDefaultStreamCurrentDevice(self.inner, &mut stream)
        })?;
        Ok(cudaStream { inner: stream })
    }

    pub fn sync_default_stream(&self, device: i32) -> Result<()> {
        faiss_rc(unsafe { ffi::faiss_GpuResources_syncDefaultStream(self.inner, device) })
    }

    pub fn sync_default_stream_current_device(&self) -> Result<()> {
        faiss_rc(unsafe { ffi::faiss_GpuResources_syncDefaultStreamCurrentDevice(self.inner) })
    }

    pub fn get_async_copy_stream_current_device(&self) -> Result<cudaStream> {
        let mut stream = null_mut();
        faiss_rc(unsafe {
            ffi::faiss_GpuResources_getAsyncCopyStreamCurrentDevice(self.inner, &mut stream)
        })?;
        Ok(cudaStream { inner: stream })
    }
}

pub trait FaissGpuResourcesProviderTrait {
    fn inner(&self) -> *mut ffi::FaissGpuResourcesProvider;
    fn get_resources(&self) -> Result<FaissGpuResources> {
        let mut resources = null_mut();
        faiss_rc(unsafe {
            ffi::faiss_GpuResourcesProvider_getResources(self.inner(), &mut resources)
        })?;
        Ok(FaissGpuResources { inner: resources })
    }
}
