use std::ptr;

use crate::bindings;
use crate::cuda::gpu_resources::GpuResourcesProvider;
use crate::error::{check_return_code, Result};
use crate::index::Index;
use crate::traits::FaissIndex;

pub struct GpuIndex {
    pub(crate) inner: *mut bindings::FaissIndex,
}

impl Drop for GpuIndex {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { bindings::faiss_GpuIndex_free(self.inner) }
        }
    }
}

impl FaissIndex for GpuIndex {
    fn inner(&self) -> *mut bindings::FaissIndex {
        self.inner
    }

    fn train(&mut self, n: i64, x: &[f32]) -> Result<()> {
        check_return_code(unsafe { bindings::faiss_Index_train(self.inner, n, x.as_ptr()) })
    }

    fn add(&mut self, n: i64, x: &[f32]) -> Result<()> {
        check_return_code(unsafe { bindings::faiss_Index_add(self.inner, n, x.as_ptr()) })
    }

    fn add_with_ids(&mut self, n: i64, x: &[f32], ids: &[i64]) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_Index_add_with_ids(self.inner, n, x.as_ptr(), ids.as_ptr())
        })
    }

    fn search(
        &self,
        n: i64,
        x: &[f32],
        k: i64,
        distances: &mut [f32],
        labels: &mut [i64],
    ) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_Index_search(
                self.inner,
                n,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            )
        })
    }

    fn range_search(
        &self,
        n: i64,
        x: &[f32],
        radius: f32,
        result: *mut bindings::FaissRangeSearchResult,
    ) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_Index_range_search(self.inner, n, x.as_ptr(), radius, result)
        })
    }

    fn reset(&mut self) -> Result<()> {
        check_return_code(unsafe { bindings::faiss_Index_reset(self.inner) })
    }

    fn reconstruct(&self, key: i64, recons: &mut [f32]) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_Index_reconstruct(self.inner, key, recons.as_mut_ptr())
        })
    }
}

pub fn index_cpu_to_gpu(
    resources: &impl GpuResourcesProvider,
    device: i32,
    index: &Index,
) -> Result<GpuIndex> {
    let mut inner = ptr::null_mut();
    check_return_code(unsafe {
        bindings::faiss_index_cpu_to_gpu(resources.inner(), device, index.inner, &mut inner)
    })?;
    Ok(GpuIndex { inner })
}

pub fn index_gpu_to_cpu(index: &GpuIndex) -> Result<Index> {
    let mut inner = ptr::null_mut();
    check_return_code(unsafe { bindings::faiss_index_gpu_to_cpu(index.inner, &mut inner) })?;
    Ok(Index { inner })
}
