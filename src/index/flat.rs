use std::ptr;

use faiss_next_sys::{self, FaissIndex, FaissIndexFlat};

use crate::error::{check_return_code, Result};
use crate::index::native::InnerPtr;
use crate::index::traits::Index;
use crate::metric::MetricType;

pub struct IndexFlat {
    inner: InnerPtr<FaissIndexFlat>,
}

impl IndexFlat {
    pub fn new(d: u32, metric: MetricType) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IndexFlat_new_with(
                &mut inner,
                d as i64,
                metric.as_native(),
            ))?;
            Ok(Self {
                inner: InnerPtr::new(inner)?,
            })
        }
    }

    pub fn new_l2(d: u32) -> Result<Self> {
        Self::new(d, MetricType::L2)
    }

    pub fn new_ip(d: u32) -> Result<Self> {
        Self::new(d, MetricType::InnerProduct)
    }

    pub fn xb(&self) -> &[f32] {
        unsafe {
            let mut ptr = ptr::null_mut();
            let mut size = 0usize;
            faiss_next_sys::faiss_IndexFlat_xb(self.inner.as_ptr(), &mut ptr, &mut size);
            if ptr.is_null() || size == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(ptr, size)
            }
        }
    }

    pub(crate) unsafe fn from_raw(ptr: *mut FaissIndexFlat) -> Self {
        Self {
            inner: InnerPtr::new(ptr).expect("ptr is null"),
        }
    }
}

impl Index for IndexFlat {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner.as_ptr() as *mut FaissIndex
    }
}

impl Drop for IndexFlat {
    fn drop(&mut self) {
        tracing::trace!("dropping IndexFlat");
        unsafe {
            faiss_next_sys::faiss_IndexFlat_free(self.inner.as_ptr());
        }
    }
}

unsafe impl Send for IndexFlat {}
unsafe impl Sync for IndexFlat {}
