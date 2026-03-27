use std::ptr;

use faiss_next_sys::{FaissGpuIndex, FaissGpuResourcesProvider, FaissIndex};

use crate::error::{check_return_code, Error, Result};
use crate::index::Index;

use super::GpuResources;

pub struct GpuIndexImpl {
    inner: *mut FaissGpuIndex,
    #[allow(dead_code)]
    resources: GpuResources,
}

impl GpuIndexImpl {
    pub fn from_cpu(index: &impl Index, resources: GpuResources, device: i32) -> Result<Self> {
        unsafe {
            let mut gpu_index = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_index_cpu_to_gpu(
                resources.inner as *mut FaissGpuResourcesProvider,
                device,
                index.inner_ptr(),
                &mut gpu_index,
            ))?;
            Ok(Self {
                inner: gpu_index,
                resources,
            })
        }
    }

    pub fn to_cpu(&self) -> Result<crate::index::IndexImpl> {
        unsafe {
            let mut cpu_index = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_index_gpu_to_cpu(
                self.inner as *mut FaissIndex,
                &mut cpu_index,
            ))?;
            crate::index::IndexImpl::from_raw(cpu_index)
        }
    }
}

impl Index for GpuIndexImpl {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner as *mut FaissIndex
    }

    fn range_search(
        &mut self,
        _q: &[f32],
        _radius: f32,
    ) -> Result<crate::result::RangeSearchResult> {
        Err(Error::unsupported(
            "range_search not implemented for GPU index",
        ))
    }
}

impl Drop for GpuIndexImpl {
    fn drop(&mut self) {
        tracing::trace!("dropping GpuIndexImpl");
        unsafe {
            faiss_next_sys::faiss_Index_free(self.inner as *mut FaissIndex);
        }
    }
}

unsafe impl Send for GpuIndexImpl {}
unsafe impl Sync for GpuIndexImpl {}
