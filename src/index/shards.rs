use std::ptr;

use faiss_next_sys::{self, FaissIndex, FaissIndexShards};

use crate::error::{check_return_code, Result};
use crate::index::native::InnerPtr;
use crate::index::traits::Index;

pub struct IndexShards {
    inner: InnerPtr<FaissIndexShards>,
}

impl IndexShards {
    pub fn new(d: u32) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IndexShards_new(&mut inner, d as i64))?;
            Ok(Self {
                inner: InnerPtr::new(inner)?,
            })
        }
    }

    pub fn add_shard(&mut self, index: impl Index) {
        unsafe {
            faiss_next_sys::faiss_IndexShards_add_shard(self.inner.as_ptr(), index.inner_ptr());
        }
        std::mem::forget(index);
    }

    pub fn at(&self, i: usize) -> *mut FaissIndex {
        unsafe { faiss_next_sys::faiss_IndexShards_at(self.inner.as_ptr(), i as i32) }
    }

    pub fn successive_ids(&self) -> bool {
        unsafe { faiss_next_sys::faiss_IndexShards_successive_ids(self.inner.as_ptr()) != 0 }
    }

    pub fn set_successive_ids(&mut self, successive: bool) {
        unsafe {
            faiss_next_sys::faiss_IndexShards_set_successive_ids(
                self.inner.as_ptr(),
                successive as i32,
            )
        }
    }
}

impl Index for IndexShards {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner.as_ptr() as *mut FaissIndex
    }
}

impl Drop for IndexShards {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexShards");
        unsafe {
            faiss_next_sys::faiss_IndexShards_free(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for IndexShards {}
unsafe impl Sync for IndexShards {}
