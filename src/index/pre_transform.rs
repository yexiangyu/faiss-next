use std::ptr;

use faiss_next_sys::{self, FaissIndex, FaissIndexPreTransform};

use crate::error::{check_return_code, Error, Result};
use crate::index::native::InnerPtr;
use crate::index::traits::Index;

pub struct IndexPreTransform {
    inner: InnerPtr<FaissIndexPreTransform>,
}

impl IndexPreTransform {
    pub fn new(index: super::IndexImpl) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IndexPreTransform_new_with(
                &mut inner,
                index.inner_ptr(),
            ))?;
            std::mem::forget(index);
            Ok(Self {
                inner: InnerPtr::new(inner)?,
            })
        }
    }

    pub fn from_index(index: super::IndexImpl) -> Result<Self> {
        unsafe {
            let pt_ptr = faiss_next_sys::faiss_IndexPreTransform_cast(index.inner_ptr());
            if pt_ptr.is_null() {
                return Err(Error::invalid_cast(
                    "IndexPreTransform",
                    "index is not a PreTransform index",
                ));
            }
            std::mem::forget(index);
            Ok(Self {
                inner: InnerPtr::new(pt_ptr)?,
            })
        }
    }

    pub fn sub_index(&self) -> *mut FaissIndex {
        unsafe { faiss_next_sys::faiss_IndexPreTransform_index(self.inner.as_ptr()) }
    }
}

impl Index for IndexPreTransform {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner.as_ptr() as *mut FaissIndex
    }
}

impl Drop for IndexPreTransform {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexPreTransform");
        unsafe {
            faiss_next_sys::faiss_IndexPreTransform_free(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for IndexPreTransform {}
unsafe impl Sync for IndexPreTransform {}
