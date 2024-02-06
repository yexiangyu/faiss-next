use std::mem::ManuallyDrop;
use std::ptr::null_mut;

use super::{FaissMetricType, Index, IndexPtr};
use crate::error::{Error, Result};
use crate::macros::{define_index_impl, faiss_rc};
use faiss_next_sys as sys;

pub fn faiss_index_factory(
    d: i32,
    description: impl AsRef<str>,
    metric: FaissMetricType,
) -> Result<FaissIndexImpl> {
    let description = description.as_ref();
    let description = std::ffi::CString::new(description).map_err(|_| Error::InvalidDescription)?;
    let mut inner = null_mut();
    faiss_rc!({ sys::faiss_index_factory(&mut inner, d, description.as_ptr(), metric) })?;
    Ok(FaissIndexImpl { inner })
}

define_index_impl!(
    /// Faiss index created by faiss_index_factory
    FaissIndexImpl,
    faiss_Index_free
);
impl FaissIndexImpl {
    pub fn from_ptr(ptr: *mut sys::FaissIndex) -> ManuallyDrop<Self> {
        ManuallyDrop::new(Self {
            inner: ptr as *mut _,
        })
    }
}
