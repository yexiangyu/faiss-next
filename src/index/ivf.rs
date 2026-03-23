use std::ptr;

use faiss_next_sys::{self, FaissIndex, FaissIndexIVF, FaissIndexIVFFlat};

use crate::error::{check_return_code, Error, Result};
use crate::index::native::InnerPtr;
use crate::index::traits::{Index, IvfIndex};

pub struct IndexIVFFlat {
    inner: InnerPtr<FaissIndexIVFFlat>,
}

impl IndexIVFFlat {
    pub fn new(quantizer: super::IndexFlat, nlist: usize) -> Result<Self> {
        let d = quantizer.d();
        let quantizer_ptr = quantizer.inner_ptr();

        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IndexIVFFlat_new_with(
                &mut inner,
                quantizer_ptr as *mut FaissIndex,
                d as usize,
                nlist,
            ))?;

            std::mem::forget(quantizer);

            Ok(Self {
                inner: InnerPtr::new(inner)?,
            })
        }
    }

    pub fn from_index(index: super::IndexImpl) -> Result<Self> {
        unsafe {
            let ivf_ptr = faiss_next_sys::faiss_IndexIVFFlat_cast(index.inner_ptr());
            if ivf_ptr.is_null() {
                return Err(Error::invalid_cast(
                    "IndexIVFFlat",
                    "index is not an IVFFlat index",
                ));
            }
            std::mem::forget(index);
            Ok(Self {
                inner: InnerPtr::new(ivf_ptr)?,
            })
        }
    }

    pub fn quantizer(&self) -> *mut FaissIndex {
        unsafe { faiss_next_sys::faiss_IndexIVFFlat_quantizer(self.inner.as_ptr()) }
    }
}

impl Index for IndexIVFFlat {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner.as_ptr() as *mut FaissIndex
    }
}

impl IvfIndex for IndexIVFFlat {
    fn nlist(&self) -> usize {
        unsafe { faiss_next_sys::faiss_IndexIVFFlat_nlist(self.inner.as_ptr()) }
    }

    fn nprobe(&self) -> usize {
        unsafe { faiss_next_sys::faiss_IndexIVFFlat_nprobe(self.inner.as_ptr()) }
    }

    fn set_nprobe(&mut self, nprobe: usize) {
        unsafe { faiss_next_sys::faiss_IndexIVFFlat_set_nprobe(self.inner.as_ptr(), nprobe) }
    }
}

impl Drop for IndexIVFFlat {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexIVFFlat");
        unsafe {
            faiss_next_sys::faiss_IndexIVFFlat_free(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for IndexIVFFlat {}
unsafe impl Sync for IndexIVFFlat {}

pub struct IndexIVF {
    inner: InnerPtr<FaissIndexIVF>,
}

impl IndexIVF {
    pub fn from_index(index: super::IndexImpl) -> Result<Self> {
        unsafe {
            let ivf_ptr = faiss_next_sys::faiss_IndexIVF_cast(index.inner_ptr());
            if ivf_ptr.is_null() {
                return Err(Error::invalid_cast("IndexIVF", "index is not an IVF index"));
            }
            std::mem::forget(index);
            Ok(Self {
                inner: InnerPtr::new(ivf_ptr)?,
            })
        }
    }
}

impl Index for IndexIVF {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner.as_ptr() as *mut FaissIndex
    }
}

impl IvfIndex for IndexIVF {
    fn nlist(&self) -> usize {
        unsafe { faiss_next_sys::faiss_IndexIVF_nlist(self.inner.as_ptr()) }
    }

    fn nprobe(&self) -> usize {
        unsafe { faiss_next_sys::faiss_IndexIVF_nprobe(self.inner.as_ptr()) }
    }

    fn set_nprobe(&mut self, nprobe: usize) {
        unsafe { faiss_next_sys::faiss_IndexIVF_set_nprobe(self.inner.as_ptr(), nprobe) }
    }
}

impl Drop for IndexIVF {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexIVF");
        unsafe {
            faiss_next_sys::faiss_IndexIVF_free(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for IndexIVF {}
unsafe impl Sync for IndexIVF {}
