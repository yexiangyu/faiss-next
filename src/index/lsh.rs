use std::ptr;

use faiss_next_sys::{self, FaissIndex, FaissIndexLSH};

use crate::error::{check_return_code, Error, Result};
use crate::index::native::InnerPtr;
use crate::index::traits::Index;

pub struct IndexLSH {
    inner: InnerPtr<FaissIndexLSH>,
}

impl IndexLSH {
    pub fn new(d: u32, nbits: u32) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IndexLSH_new(
                &mut inner,
                d as i64,
                nbits as i32,
            ))?;
            Ok(Self {
                inner: InnerPtr::new(inner)?,
            })
        }
    }

    pub fn from_index(index: super::IndexImpl) -> Result<Self> {
        unsafe {
            let lsh_ptr = faiss_next_sys::faiss_IndexLSH_cast(index.inner_ptr());
            if lsh_ptr.is_null() {
                return Err(Error::invalid_cast("IndexLSH", "index is not an LSH index"));
            }
            std::mem::forget(index);
            Ok(Self {
                inner: InnerPtr::new(lsh_ptr)?,
            })
        }
    }

    pub fn nbits(&self) -> u32 {
        unsafe { faiss_next_sys::faiss_IndexLSH_nbits(self.inner.as_ptr()) as u32 }
    }

    pub fn code_size(&self) -> u32 {
        unsafe { faiss_next_sys::faiss_IndexLSH_code_size(self.inner.as_ptr()) as u32 }
    }
}

impl Index for IndexLSH {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner.as_ptr() as *mut FaissIndex
    }
}

impl Drop for IndexLSH {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexLSH");
        unsafe {
            faiss_next_sys::faiss_IndexLSH_free(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for IndexLSH {}
unsafe impl Sync for IndexLSH {}
