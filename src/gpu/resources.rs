use std::ptr;

use faiss_next_sys::FaissGpuResources;

use crate::error::{check_return_code, Result};

pub struct GpuResources {
    inner: *mut FaissGpuResources,
}

impl GpuResources {
    pub fn new() -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_GpuResources_new(&mut inner))?;
            Ok(Self { inner })
        }
    }

    pub fn no_temp_memory(&mut self) -> Result<()> {
        check_return_code(unsafe { faiss_next_sys::faiss_GpuResources_no_temp_memory(self.inner) })
    }

    pub fn set_temp_memory(&mut self, size: usize) -> Result<()> {
        check_return_code(unsafe {
            faiss_next_sys::faiss_GpuResources_set_temp_memory(self.inner, size)
        })
    }

    pub fn set_pinned_memory(&mut self, size: usize) -> Result<()> {
        check_return_code(unsafe {
            faiss_next_sys::faiss_GpuResources_set_pinned_memory(self.inner, size)
        })
    }

    pub fn initialize_for_device(&mut self, device: i32) -> Result<()> {
        check_return_code(unsafe {
            faiss_next_sys::faiss_GpuResources_initializeForDevice(self.inner, device)
        })
    }
}

impl Default for GpuResources {
    fn default() -> Self {
        Self::new().expect("failed to create GpuResources")
    }
}

impl Drop for GpuResources {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                faiss_next_sys::faiss_GpuResources_free(self.inner);
            }
        }
    }
}

unsafe impl Send for GpuResources {}
unsafe impl Sync for GpuResources {}
