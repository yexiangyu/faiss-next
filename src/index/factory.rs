use std::ffi::CString;
use std::ptr::null_mut;
use tracing::*;

use faiss_next_sys as sys;

use crate::error::{Error, Result};
use crate::index::{impl_index, IndexTrait};
use crate::macros::rc;
use crate::metric::MetricType;

pub struct IndexImpl {
    inner: *mut sys::FaissIndex,
}

impl_index!(IndexImpl);

impl IndexImpl {
    pub fn new(inner: *mut sys::FaissIndex) -> Self {
        let r = Self { inner };
        trace!(?r, "new");
        r
    }
}

pub fn index_factory(
    d: usize,
    description: impl AsRef<str>,
    metric: MetricType,
) -> Result<IndexImpl> {
    let mut inner = null_mut();
    let desc = description.as_ref();
    let description =
        CString::new(desc).map_err(|_| Error::InvalidDescription { desc: desc.into() })?;
    rc!({ sys::faiss_index_factory(&mut inner, d as i32, description.as_ptr(), metric.into()) })?;
    let r = IndexImpl::new(inner);
    trace!(?desc, ?d, ?r, "index_factory");
    Ok(r)
}
