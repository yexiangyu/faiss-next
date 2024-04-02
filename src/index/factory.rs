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
        Self { inner }
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
    let r = IndexImpl { inner };
    trace!(?r, "index_factory");
    Ok(r)
}

#[cfg(test)]
#[test]
fn test_index_factory_ok() -> Result<()> {
    std::env::set_var("RUST_LOG", "trace");
    let _ = tracing_subscriber::fmt::try_init();

    use crate::index::SearchParameters;
    use ndarray::{s, Array2};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    let d = 128;
    let n = 1024;
    let k = 1;
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
    index.add(base, Some(ids))?;
    trace!(?index);
    let mut distances = vec![0.0f32; k as usize];
    let mut labels = vec![0i64; k as usize];
    index.search(
        query,
        1,
        &mut distances,
        &mut labels,
        Option::<SearchParameters>::None,
    )?;
    info!("distances={:?}, labels={:?}", distances, labels);
    Ok(())
}
