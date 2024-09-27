use std::ptr::null_mut;

use crate::{error::*, macros::*};
use faiss_next_sys as ffi;

use super::{gpu_resources::cudaStream, prelude::FaissGpuResourcesProviderTrait};

#[derive(Debug)]
pub struct FaissStandardGpuResources {
    pub inner: *mut ffi::FaissGpuResourcesProvider,
}
impl_faiss_drop!(FaissStandardGpuResources, faiss_GpuResourcesProvider_free);

impl FaissGpuResourcesProviderTrait for FaissStandardGpuResources {
    fn inner(&self) -> *mut ffi::FaissGpuResourcesProvider {
        self.inner
    }
}

impl FaissStandardGpuResources {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_StandardGpuResources_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn no_temp_memory(&mut self) -> Result<()> {
        faiss_rc(unsafe { ffi::faiss_StandardGpuResources_noTempMemory(self.inner) })
    }
    pub fn set_temp_memory(&mut self, size: usize) -> Result<()> {
        faiss_rc(unsafe { ffi::faiss_StandardGpuResources_setTempMemory(self.inner, size) })
    }
    pub fn set_pinned_memory(&mut self, size: usize) -> Result<()> {
        faiss_rc(unsafe { ffi::faiss_StandardGpuResources_setPinnedMemory(self.inner, size) })
    }
    pub fn set_default_stream(&mut self, device: i32, stream: cudaStream) -> Result<()> {
        faiss_rc(unsafe {
            ffi::faiss_StandardGpuResources_setDefaultStream(self.inner, device, stream.inner)
        })
    }
    pub fn set_default_null_stream_all_devices(&mut self) -> Result<()> {
        faiss_rc(unsafe {
            ffi::faiss_StandardGpuResources_setDefaultNullStreamAllDevices(self.inner)
        })
    }
}
