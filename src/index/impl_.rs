use faiss_next_sys::{self, FaissIndex};

use crate::error::{Error, Result};
use crate::index::native::InnerPtr;
use crate::index::traits::Index;

pub struct IndexImpl {
    inner: InnerPtr<FaissIndex>,
}

impl IndexImpl {
    pub(crate) fn from_raw(ptr: *mut FaissIndex) -> Result<Self> {
        Ok(Self {
            inner: InnerPtr::new(ptr)?,
        })
    }

    pub fn into_flat(self) -> Result<super::IndexFlat> {
        unsafe {
            let flat_ptr = faiss_next_sys::faiss_IndexFlat_cast(self.inner.as_ptr());
            if flat_ptr.is_null() {
                return Err(Error::invalid_cast(
                    "IndexFlat",
                    "index is not a flat index",
                ));
            }
            std::mem::forget(self);
            Ok(super::IndexFlat::from_raw(flat_ptr))
        }
    }
}

impl Index for IndexImpl {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner.as_ptr()
    }
}

impl Drop for IndexImpl {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexImpl");
        unsafe {
            faiss_next_sys::faiss_Index_free(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for IndexImpl {}
unsafe impl Sync for IndexImpl {}
