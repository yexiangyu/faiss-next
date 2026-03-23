use std::ptr;

use faiss_next_sys::{self, FaissIndex, FaissIndexReplicas};

use crate::error::{check_return_code, Result};
use crate::index::native::InnerPtr;
use crate::index::traits::Index;

pub struct IndexReplicas {
    inner: InnerPtr<FaissIndexReplicas>,
}

impl IndexReplicas {
    pub fn new(d: u32) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IndexReplicas_new(
                &mut inner, d as i64,
            ))?;
            Ok(Self {
                inner: InnerPtr::new(inner)?,
            })
        }
    }

    pub fn add_replica(&mut self, index: impl Index) {
        unsafe {
            faiss_next_sys::faiss_IndexReplicas_add_replica(self.inner.as_ptr(), index.inner_ptr());
        }
        std::mem::forget(index);
    }

    pub fn at(&self, i: usize) -> *mut FaissIndex {
        unsafe { faiss_next_sys::faiss_IndexReplicas_at(self.inner.as_ptr(), i as i32) }
    }
}

impl Index for IndexReplicas {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner.as_ptr() as *mut FaissIndex
    }
}

impl Drop for IndexReplicas {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexReplicas");
        unsafe {
            faiss_next_sys::faiss_IndexReplicas_free(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for IndexReplicas {}
unsafe impl Sync for IndexReplicas {}
