use std::ffi::CString;
use std::ptr;

use crate::bindings::FaissMetricType;
use crate::error::{check_return_code, Result};
use crate::index::Index;

pub fn index_factory(d: i32, description: &str, metric: FaissMetricType) -> Result<Index> {
    let c_description = CString::new(description)?;
    let mut inner = ptr::null_mut();
    check_return_code(unsafe {
        crate::bindings::faiss_index_factory(&mut inner, d, c_description.as_ptr(), metric)
    })?;
    Ok(Index { inner })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    fn generate_random_vectors(n: usize, d: usize) -> Array2<f32> {
        Array2::random((n, d), Uniform::new(-1.0f32, 1.0f32))
    }

    #[test]
    fn test_index_factory_flat() {
        let d = 32;
        let n = 100;
        let k = 5;

        let mut index = index_factory(d as i32, "Flat", FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);
        assert_eq!(index.d(), d as i32);

        let query = data.row(50).to_owned();
        let mut distances = vec![0.0f32; k];
        let mut labels = vec![-1i64; k];

        index
            .search(
                1,
                query.as_slice().unwrap(),
                k as i64,
                &mut distances,
                &mut labels,
            )
            .unwrap();

        assert_eq!(labels[0], 50);
    }

    #[test]
    fn test_index_factory_ivf_flat() {
        let d = 32;
        let n = 1000;
        let k = 10;
        let nlist = 10;

        let mut index = index_factory(
            d as i32,
            &format!("IVF{},Flat", nlist),
            FaissMetricType::METRIC_L2,
        )
        .unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.train(n as i64, &data_slice).unwrap();
        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);

        let query = data.row(100).to_owned();
        let mut distances = vec![0.0f32; k];
        let mut labels = vec![-1i64; k];

        index
            .search(
                1,
                query.as_slice().unwrap(),
                k as i64,
                &mut distances,
                &mut labels,
            )
            .unwrap();

        assert!(labels[0] >= 0);
    }

    #[test]
    fn test_index_factory_pq() {
        let d = 32;
        let n = 500;
        let k = 5;

        let mut index = index_factory(d as i32, "PQ16", FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.train(n as i64, &data_slice).unwrap();
        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);

        let query = data.row(42).to_owned();
        let mut distances = vec![0.0f32; k];
        let mut labels = vec![-1i64; k];

        index
            .search(
                1,
                query.as_slice().unwrap(),
                k as i64,
                &mut distances,
                &mut labels,
            )
            .unwrap();

        assert!(labels[0] >= 0);
    }

    #[test]
    fn test_index_factory_ivf_pq() {
        let d = 32;
        let n = 1000;
        let k = 5;

        let mut index = index_factory(d as i32, "IVF10,PQ8", FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.train(n as i64, &data_slice).unwrap();
        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);

        let query = data.row(200).to_owned();
        let mut distances = vec![0.0f32; k];
        let mut labels = vec![-1i64; k];

        index
            .search(
                1,
                query.as_slice().unwrap(),
                k as i64,
                &mut distances,
                &mut labels,
            )
            .unwrap();

        assert!(labels[0] >= 0);
    }

    #[test]
    fn test_index_factory_hnsw() {
        let d = 16;
        let n = 200;
        let k = 5;

        let mut index = index_factory(d as i32, "HNSW32", FaissMetricType::METRIC_L2).unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);

        let query = data.row(100).to_owned();
        let mut distances = vec![0.0f32; k];
        let mut labels = vec![-1i64; k];

        index
            .search(
                1,
                query.as_slice().unwrap(),
                k as i64,
                &mut distances,
                &mut labels,
            )
            .unwrap();

        assert_eq!(labels[0], 100);
    }
}
