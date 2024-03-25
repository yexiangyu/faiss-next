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

pub fn index_factory(
    d: i32,
    description: impl AsRef<str>,
    metric: MetricType,
) -> Result<IndexImpl> {
    let mut inner = null_mut();
    let desc = description.as_ref();
    let description =
        CString::new(desc).map_err(|_| Error::InvalidDescription { desc: desc.into() })?;
    rc!({ sys::faiss_index_factory(&mut inner, d, description.as_ptr(), metric) })?;
    trace!(
        "create IndexImpl inner={:?}, d={}, description={}",
        inner,
        d,
        desc
    );
    Ok(IndexImpl { inner })
}

#[cfg(test)]
#[test]
fn test_index_factory_ok() -> Result<()> {
    use ndarray::{s, Array2};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    std::env::set_var("RUST_LOG", "trace");
    let _ = tracing_subscriber::fmt::try_init();
    let d = 128;
    let n = 1024;
    let ids = (0..n).map(|i| i as i64).collect::<Vec<_>>();
    let base = Array2::random([n, d], Uniform::new(-1.0f32, 1.0f32));
    let query = base.slice(s![42, ..]);
    let query = query
        .as_slice_memory_order()
        .ok_or(Error::NotStandardLayout)?;
    let base = base
        .as_slice_memory_order()
        .ok_or(Error::NotStandardLayout)?;
    let mut index = index_factory(128, "IDMap,Flat", MetricType::METRIC_L2)?;
    index.add(base, Option::<&[i64]>::None)?;
    trace!(?index);
    Ok(())
}
