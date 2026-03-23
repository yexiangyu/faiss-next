use faiss_next_sys::{self, FaissIndexBinary};

use crate::error::Result;
use crate::index::native::InnerPtr;
use crate::index::traits::BinaryIndex;

pub struct IndexBinary {
    inner: InnerPtr<FaissIndexBinary>,
}

impl IndexBinary {
    pub fn from_raw(ptr: *mut FaissIndexBinary) -> Result<Self> {
        Ok(Self {
            inner: InnerPtr::new(ptr)?,
        })
    }
}

impl BinaryIndex for IndexBinary {
    fn inner_ptr(&self) -> *mut FaissIndexBinary {
        self.inner.as_ptr()
    }
}

impl Drop for IndexBinary {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexBinary");
        unsafe {
            faiss_next_sys::faiss_IndexBinary_free(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for IndexBinary {}
unsafe impl Sync for IndexBinary {}
