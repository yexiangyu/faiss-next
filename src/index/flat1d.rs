use std::ptr;

use faiss_next_sys::{self, FaissIndex, FaissIndexFlat1D};

use crate::error::{check_return_code, Error, Result};
use crate::index::native::InnerPtr;
use crate::index::traits::Index;

pub struct IndexFlat1D {
    inner: InnerPtr<FaissIndexFlat1D>,
}

impl IndexFlat1D {
    pub fn new() -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IndexFlat1D_new(&mut inner))?;
            Ok(Self {
                inner: InnerPtr::new(inner)?,
            })
        }
    }

    pub fn with_contiguous() -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IndexFlat1D_new_with(&mut inner, 1))?;
            Ok(Self {
                inner: InnerPtr::new(inner)?,
            })
        }
    }

    pub fn from_index(index: super::IndexImpl) -> Result<Self> {
        unsafe {
            let flat1d_ptr = faiss_next_sys::faiss_IndexFlat1D_cast(index.inner_ptr());
            if flat1d_ptr.is_null() {
                return Err(Error::invalid_cast(
                    "IndexFlat1D",
                    "index is not a Flat1D index",
                ));
            }
            std::mem::forget(index);
            Ok(Self {
                inner: InnerPtr::new(flat1d_ptr)?,
            })
        }
    }

    pub fn update_permutation(&mut self) -> Result<()> {
        check_return_code(unsafe {
            faiss_next_sys::faiss_IndexFlat1D_update_permutation(self.inner.as_ptr())
        })
    }
}

impl Index for IndexFlat1D {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner.as_ptr() as *mut FaissIndex
    }
}

impl Drop for IndexFlat1D {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexFlat1D");
        unsafe {
            faiss_next_sys::faiss_IndexFlat1D_free(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for IndexFlat1D {}
unsafe impl Sync for IndexFlat1D {}
