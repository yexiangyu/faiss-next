use std::ptr;

use faiss_next_sys::{self, FaissIndex, FaissIndexRefineFlat};

use crate::error::{check_return_code, Error, Result};
use crate::index::native::InnerPtr;
use crate::index::traits::Index;

pub struct IndexRefineFlat {
    inner: InnerPtr<FaissIndexRefineFlat>,
}

impl IndexRefineFlat {
    pub fn new(base_index: super::IndexImpl) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IndexRefineFlat_new(
                &mut inner,
                base_index.inner_ptr(),
            ))?;
            std::mem::forget(base_index);
            Ok(Self {
                inner: InnerPtr::new(inner)?,
            })
        }
    }

    pub fn from_index(index: super::IndexImpl) -> Result<Self> {
        unsafe {
            let rf_ptr = faiss_next_sys::faiss_IndexRefineFlat_cast(index.inner_ptr());
            if rf_ptr.is_null() {
                return Err(Error::invalid_cast(
                    "IndexRefineFlat",
                    "index is not a RefineFlat index",
                ));
            }
            std::mem::forget(index);
            Ok(Self {
                inner: InnerPtr::new(rf_ptr)?,
            })
        }
    }

    pub fn k_factor(&self) -> f32 {
        unsafe { faiss_next_sys::faiss_IndexRefineFlat_k_factor(self.inner.as_ptr()) }
    }

    pub fn set_k_factor(&mut self, k_factor: f32) {
        unsafe { faiss_next_sys::faiss_IndexRefineFlat_set_k_factor(self.inner.as_ptr(), k_factor) }
    }
}

impl Index for IndexRefineFlat {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner.as_ptr() as *mut FaissIndex
    }
}

impl Drop for IndexRefineFlat {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexRefineFlat");
        unsafe {
            faiss_next_sys::faiss_IndexRefineFlat_free(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for IndexRefineFlat {}
unsafe impl Sync for IndexRefineFlat {}
