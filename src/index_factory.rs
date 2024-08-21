use faiss_next_sys::{self as ffi, FaissMetricType};

use crate::{error::*, index::FaissIndexOwned};

/// build index with factory function
/// ```rust
/// use faiss_next::prelude::*;
/// use itertools::Itertools;
/// use ndarray::{Array2, s};
/// use ndarray_rand::{rand_distr::Uniform, RandomExt};
///
/// let index = faiss_index_factory(128, "Flat", FaissMetricType::METRIC_L2).unwrap();
/// let mut index = FaissIndexFlat::downcast(index).unwrap();
/// let base = Array2::<f32>::random((1024, 128), Uniform::new(-1.0, 1.0));
///
/// let query = base.slice(s![42..43, ..]);
///
/// index.add(base.as_slice().unwrap()).unwrap();
///
/// let mut distances = vec![0.0];
/// let mut labels = vec![-1];
///
/// index.search(query.as_slice().unwrap(), 1, &mut distances, &mut labels).unwrap();
/// assert_eq!(labels, &[42]);
/// assert_eq!(index.xb().len(), 1024 * 128);
/// ```
pub fn faiss_index_factory(
    d: i32,
    description: impl AsRef<str>,
    metric: FaissMetricType,
) -> Result<FaissIndexOwned> {
    let description = description.as_ref();
    let description = std::ffi::CString::new(description).unwrap();
    let mut index = std::ptr::null_mut();
    crate::error::faiss_rc(unsafe {
        ffi::faiss_index_factory(&mut index, d, description.as_ptr(), metric)
    })?;
    Ok(FaissIndexOwned { inner: index })
}
