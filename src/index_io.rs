use std::ptr;

use crate::bindings;
use crate::error::{check_return_code, Result};
use crate::index::Index;

pub fn write_index(index: &Index, path: &str) -> Result<()> {
    let c_path = std::ffi::CString::new(path)?;
    check_return_code(unsafe { bindings::faiss_write_index_fname(index.inner, c_path.as_ptr()) })?;
    Ok(())
}

pub fn read_index(path: &str) -> Result<Index> {
    let c_path = std::ffi::CString::new(path)?;
    let mut inner = ptr::null_mut();
    check_return_code(unsafe { bindings::faiss_read_index_fname(c_path.as_ptr(), 0, &mut inner) })?;
    Ok(Index { inner })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::FaissMetricType;
    use crate::index_factory::index_factory;
    use ndarray::Array2;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use std::path::Path;

    fn generate_random_vectors(n: usize, d: usize) -> Array2<f32> {
        Array2::random((n, d), Uniform::new(-1.0f32, 1.0f32))
    }

    fn cleanup_test_file(path: &str) {
        let p = Path::new(path);
        if p.exists() {
            std::fs::remove_file(p).ok();
        }
    }

    #[test]
    fn test_write_read_index_flat() {
        let d = 32;
        let n = 100;
        let k = 5;
        let path = "/tmp/test_index_flat.bin";

        cleanup_test_file(path);

        let mut index = index_factory(d as i32, "Flat", FaissMetricType::METRIC_L2).unwrap();
        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();
        index.add(n as i64, &data_slice).unwrap();

        write_index(&index, path).unwrap();
        assert!(Path::new(path).exists());

        let loaded_index = read_index(path).unwrap();
        assert_eq!(loaded_index.ntotal(), n as i64);
        assert_eq!(loaded_index.d(), d as i32);

        let query = data.row(42).to_owned();
        let mut distances = vec![0.0f32; k];
        let mut labels = vec![-1i64; k];

        loaded_index
            .search(
                1,
                query.as_slice().unwrap(),
                k as i64,
                &mut distances,
                &mut labels,
            )
            .unwrap();

        assert_eq!(labels[0], 42);

        cleanup_test_file(path);
    }

    #[test]
    fn test_write_read_preserves_search() {
        let d = 16;
        let n = 50;
        let k = 3;
        let path = "/tmp/test_index_search.bin";

        cleanup_test_file(path);

        let mut index = index_factory(d as i32, "Flat", FaissMetricType::METRIC_L2).unwrap();
        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();
        index.add(n as i64, &data_slice).unwrap();

        let query = data.row(10).to_owned();
        let mut distances1 = vec![0.0f32; k];
        let mut labels1 = vec![-1i64; k];
        index
            .search(
                1,
                query.as_slice().unwrap(),
                k as i64,
                &mut distances1,
                &mut labels1,
            )
            .unwrap();

        write_index(&index, path).unwrap();

        let loaded_index = read_index(path).unwrap();
        let mut distances2 = vec![0.0f32; k];
        let mut labels2 = vec![-1i64; k];
        loaded_index
            .search(
                1,
                query.as_slice().unwrap(),
                k as i64,
                &mut distances2,
                &mut labels2,
            )
            .unwrap();

        for i in 0..k {
            assert_eq!(labels1[i], labels2[i]);
            assert!((distances1[i] - distances2[i]).abs() < 1e-6);
        }

        cleanup_test_file(path);
    }
}
