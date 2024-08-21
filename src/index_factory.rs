use faiss_next_sys::{self as ffi, FaissMetricType};

use crate::{error::*, index::FaissIndexOwned};

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

#[cfg(test)]
#[test]
fn test_index_ok() -> Result<()> {
    use crate::traits::FaissIndexTrait;
    use itertools::Itertools;
    use ndarray::Array2;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    let _ = dotenv::dotenv();
    let _ = tracing_subscriber::fmt::try_init();
    let d = 128;
    let n = 1024;

    let base = Array2::<f32>::random((n, d), Uniform::new(-1.0, 1.0));
    let query = base.slice(ndarray::s![42..43, ..]);

    let mut index = faiss_index_factory(d as i32, "Flat,IDMap", FaissMetricType::METRIC_L2)?;
    index.add_with_ids(
        base.as_slice().unwrap(),
        (100..100 + n as i64).collect_vec(),
    )?;
    let mut distances = vec![0.0];
    let mut labels = vec![-1];
    index.search(query.as_slice().unwrap(), 1, &mut distances, &mut labels)?;
    tracing::info!(?distances, ?labels);
    Ok(())
}
