use crate::error::{Error, Result};
use crate::rc;
use faiss_next_sys as sys;
use std::ffi::CString;
use std::ptr::{addr_of_mut, null_mut};
use tracing::*;

use super::common::{impl_index_drop, impl_index_trait};
use super::{common::FaissIndexTrait, metric::FaissMetricType};

pub struct FaissIndexImpl {
    inner: *mut sys::FaissIndex,
}

impl_index_drop!(FaissIndexImpl, faiss_Index_free);
impl_index_trait!(FaissIndexImpl);

impl From<*mut sys::FaissIndex> for FaissIndexImpl {
    fn from(inner: *mut sys::FaissIndex) -> Self {
        Self { inner }
    }
}

pub struct FaissIndexBorrowed {
    inner: *mut sys::FaissIndex,
}

impl_index_trait!(FaissIndexBorrowed);

impl From<*mut sys::FaissIndex> for FaissIndexBorrowed {
    fn from(inner: *mut sys::FaissIndex) -> Self {
        Self { inner }
    }
}

pub fn faiss_index_factory(
    description: &str,
    d: i32,
    metric: FaissMetricType,
) -> Result<FaissIndexImpl> {
    let mut inner = null_mut();
    let description_ = CString::new(description).map_err(|_| Error::InvalidIndexDescription)?;
    rc!({ sys::faiss_index_factory(addr_of_mut!(inner), d, description_.as_ptr(), metric) })?;
    trace!(
        "create faiss index with factory inner={:?}, d={}, description={}, metric={:?}",
        inner,
        d,
        description,
        metric
    );
    Ok(FaissIndexImpl { inner })
}
