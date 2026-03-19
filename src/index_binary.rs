use std::ffi::CString;
use std::ptr;

use crate::bindings;
use crate::error::{check_return_code, Result};
use crate::macros::*;
use crate::traits::FaissIndexBinary;

pub struct IndexBinary {
    pub(crate) inner: *mut bindings::FaissIndexBinary,
}

impl_faiss_drop!(IndexBinary, faiss_IndexBinary_free);
impl_index_binary_common!(IndexBinary);

impl FaissIndexBinary for IndexBinary {
    fn inner(&self) -> *mut bindings::FaissIndexBinary {
        self.inner
    }

    fn train(&mut self, n: i64, x: &[u8]) -> Result<()> {
        check_return_code(unsafe { bindings::faiss_IndexBinary_train(self.inner, n, x.as_ptr()) })
    }

    fn add(&mut self, n: i64, x: &[u8]) -> Result<()> {
        check_return_code(unsafe { bindings::faiss_IndexBinary_add(self.inner, n, x.as_ptr()) })
    }

    fn search(
        &self,
        n: i64,
        x: &[u8],
        k: i64,
        distances: &mut [i32],
        labels: &mut [i64],
    ) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_IndexBinary_search(
                self.inner,
                n,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            )
        })
    }

    fn reset(&mut self) -> Result<()> {
        check_return_code(unsafe { bindings::faiss_IndexBinary_reset(self.inner) })
    }
}

impl IndexBinary {
    pub fn new_flat(d: i32) -> Result<Self> {
        let description = CString::new("BFlat")?;
        let mut inner = ptr::null_mut();
        let rc = crate::extension::faiss_index_binary_factory(d, description.as_ptr(), &mut inner);
        if rc != 0 {
            return Err(crate::error::Error::Faiss {
                code: rc,
                message: "Failed to create binary index".to_string(),
            });
        }
        Ok(Self { inner })
    }

    pub fn new_hash(d: i32, b: i32) -> Result<Self> {
        let description = CString::new(format!("BHash{}", b))?;
        let mut inner = ptr::null_mut();
        let rc = crate::extension::faiss_index_binary_factory(d, description.as_ptr(), &mut inner);
        if rc != 0 {
            return Err(crate::error::Error::Faiss {
                code: rc,
                message: "Failed to create binary hash index".to_string(),
            });
        }
        Ok(Self { inner })
    }

    pub fn new_multi_hash(d: i32, nhash: i32, b: i32) -> Result<Self> {
        let description = CString::new(format!("BHash{}x{}", nhash, b))?;
        let mut inner = ptr::null_mut();
        let rc = crate::extension::faiss_index_binary_factory(d, description.as_ptr(), &mut inner);
        if rc != 0 {
            return Err(crate::error::Error::Faiss {
                code: rc,
                message: "Failed to create binary multi-hash index".to_string(),
            });
        }
        Ok(Self { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    fn generate_random_binary_vectors(n: usize, d_bytes: usize) -> Array2<u8> {
        Array2::random((n, d_bytes), Uniform::new(0u8, 255u8))
    }

    #[test]
    fn test_index_binary_flat() {
        let d = 64;
        let d_bytes = d / 8;
        let n = 100;
        let k = 5;

        let mut index = IndexBinary::new_flat(d as i32).unwrap();

        let data = generate_random_binary_vectors(n, d_bytes);
        let data_slice: Vec<u8> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);

        let query = data.row(50).to_owned();
        let mut distances = vec![0i32; k];
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
        assert_eq!(distances[0], 0);
    }

    #[test]
    #[ignore = "IndexBinaryFlat does not support custom IDs. Use IndexBinaryIDMap wrapper instead."]
    fn test_index_binary_add_with_ids() {
        let d = 32;
        let d_bytes = d / 8;
        let n = 50;

        let mut index = IndexBinary::new_flat(d as i32).unwrap();

        let data = generate_random_binary_vectors(n, d_bytes);
        let data_slice: Vec<u8> = data.iter().copied().collect();
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();

        index.add_with_ids(n as i64, &data_slice, &ids).unwrap();
        assert_eq!(index.ntotal(), n as i64);

        let query = data.row(25).to_owned();
        let mut distances = vec![0i32; 1];
        let mut labels = vec![-1i64; 1];

        index
            .search(1, query.as_slice().unwrap(), 1, &mut distances, &mut labels)
            .unwrap();

        assert_eq!(labels[0], 1025);
    }

    #[test]
    fn test_index_binary_reset() {
        let d = 16;
        let d_bytes = d / 8;
        let n = 30;

        let mut index = IndexBinary::new_flat(d as i32).unwrap();

        let data = generate_random_binary_vectors(n, d_bytes);
        let data_slice: Vec<u8> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);

        index.reset().unwrap();
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn test_index_binary_multiple_queries() {
        let d = 64;
        let d_bytes = d / 8;
        let n = 200;
        let nq = 10;
        let k = 5;

        let mut index = IndexBinary::new_flat(d as i32).unwrap();

        let data = generate_random_binary_vectors(n, d_bytes);
        let data_slice: Vec<u8> = data.iter().copied().collect();

        index.add(n as i64, &data_slice).unwrap();

        let queries = data.slice(ndarray::s![0..nq, ..]).to_owned();
        let queries_slice: Vec<u8> = queries.iter().copied().collect();

        let mut distances = vec![0i32; nq * k];
        let mut labels = vec![-1i64; nq * k];

        index
            .search(
                nq as i64,
                &queries_slice,
                k as i64,
                &mut distances,
                &mut labels,
            )
            .unwrap();

        for i in 0..nq {
            assert_eq!(labels[i * k], i as i64);
        }
    }

    #[test]
    fn test_index_binary_hash() {
        let d = 64;
        let d_bytes = d / 8;
        let n = 100;
        let k = 5;
        let b = 16;

        let mut index = IndexBinary::new_hash(d as i32, b).unwrap();

        let data = generate_random_binary_vectors(n, d_bytes);
        let data_slice: Vec<u8> = data.iter().copied().collect();

        index.train(n as i64, &data_slice).unwrap();
        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);

        let query = data.row(42).to_owned();
        let mut distances = vec![0i32; k];
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
}
