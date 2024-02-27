use super::owned::FaissIndexOwned;
use super::FaissMetricType;
use crate::error::{Error, Result};
use crate::macros::rc;
use faiss_next_sys as sys;
use std::{ffi::CString, ptr::null_mut};

pub fn faiss_index_factory(
    d: i32,
    description: impl AsRef<str>,
    metric: FaissMetricType,
) -> Result<FaissIndexOwned> {
    let description = description.as_ref();
    let description = CString::new(description).map_err(|_| Error::InvalidDescription)?;
    let mut inner = null_mut();
    rc!({ sys::faiss_index_factory(&mut inner, d, description.as_ptr(), metric) })?;
    Ok(FaissIndexOwned { inner })
}
