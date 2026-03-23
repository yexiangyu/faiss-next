use std::ptr;

use faiss_next_sys::{self, FaissIndex, FaissIndexIDMap};

use crate::error::{check_return_code, Error, Result};
use crate::index::native::InnerPtr;
use crate::index::traits::Index;

pub struct IndexIDMap {
    inner: InnerPtr<FaissIndexIDMap>,
}

impl IndexIDMap {
    pub fn new(index: super::IndexImpl) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IndexIDMap_new(
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
            let idmap_ptr = faiss_next_sys::faiss_IndexIDMap_cast(index.inner_ptr());
            if idmap_ptr.is_null() {
                return Err(Error::invalid_cast(
                    "IndexIDMap",
                    "index is not an IndexIDMap",
                ));
            }
            std::mem::forget(index);
            Ok(Self {
                inner: InnerPtr::new(idmap_ptr)?,
            })
        }
    }

    pub fn sub_index(&self) -> *mut FaissIndex {
        unsafe { faiss_next_sys::faiss_IndexIDMap_sub_index(self.inner.as_ptr()) }
    }
}

impl Index for IndexIDMap {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner.as_ptr() as *mut FaissIndex
    }
}

impl Drop for IndexIDMap {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexIDMap");
        unsafe {
            faiss_next_sys::faiss_Index_free(self.inner.as_ptr() as *mut FaissIndex);
        }
    }
}

unsafe impl Send for IndexIDMap {}
unsafe impl Sync for IndexIDMap {}
