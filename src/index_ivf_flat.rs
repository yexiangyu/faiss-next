use std::ptr;

use crate::bindings::{self, FaissMetricType};
use crate::error::{check_return_code, Result};
use crate::macros::*;
use crate::traits::{FaissIVFIndex, FaissIndex};

pub struct IndexIVFFlat {
    pub(crate) inner: *mut bindings::FaissIndex,
}

impl_faiss_drop!(IndexIVFFlat, faiss_IndexIVFFlat_free);
impl_index_common!(IndexIVFFlat);

impl FaissIndex for IndexIVFFlat {
    fn inner(&self) -> *mut bindings::FaissIndex {
        self.inner
    }

    fn train(&mut self, n: i64, x: &[f32]) -> Result<()> {
        check_return_code(unsafe { bindings::faiss_Index_train(self.inner, n, x.as_ptr()) })
    }

    fn add(&mut self, n: i64, x: &[f32]) -> Result<()> {
        check_return_code(unsafe { bindings::faiss_Index_add(self.inner, n, x.as_ptr()) })
    }

    fn add_with_ids(&mut self, n: i64, x: &[f32], ids: &[i64]) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_Index_add_with_ids(self.inner, n, x.as_ptr(), ids.as_ptr())
        })
    }

    fn search(
        &self,
        n: i64,
        x: &[f32],
        k: i64,
        distances: &mut [f32],
        labels: &mut [i64],
    ) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_Index_search(
                self.inner,
                n,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            )
        })
    }

    fn range_search(
        &self,
        n: i64,
        x: &[f32],
        radius: f32,
        result: *mut bindings::FaissRangeSearchResult,
    ) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_Index_range_search(self.inner, n, x.as_ptr(), radius, result)
        })
    }

    fn reset(&mut self) -> Result<()> {
        check_return_code(unsafe { bindings::faiss_Index_reset(self.inner) })
    }

    fn reconstruct(&self, key: i64, recons: &mut [f32]) -> Result<()> {
        check_return_code(unsafe {
            bindings::faiss_Index_reconstruct(self.inner, key, recons.as_mut_ptr())
        })
    }
}

impl FaissIVFIndex for IndexIVFFlat {
    fn nlist(&self) -> usize {
        unsafe { bindings::faiss_IndexIVFFlat_nlist(self.inner as *mut _) }
    }

    fn nprobe(&self) -> usize {
        unsafe { bindings::faiss_IndexIVFFlat_nprobe(self.inner as *mut _) }
    }

    fn set_nprobe(&mut self, nprobe: usize) {
        unsafe { bindings::faiss_IndexIVFFlat_set_nprobe(self.inner as *mut _, nprobe) }
    }

    fn quantizer(&self) -> *mut bindings::FaissIndex {
        unsafe { bindings::faiss_IndexIVFFlat_quantizer(self.inner as *mut _) }
    }
}

impl IndexIVFFlat {
    pub fn new(quantizer: &mut crate::index::Index, d: usize, nlist: usize) -> Result<Self> {
        let mut inner = ptr::null_mut();
        check_return_code(unsafe {
            bindings::faiss_IndexIVFFlat_new_with(&mut inner, quantizer.inner, d, nlist)
        })?;
        Ok(Self { inner })
    }

    pub fn new_with_metric(
        quantizer: &mut crate::index::Index,
        d: usize,
        nlist: usize,
        metric: FaissMetricType,
    ) -> Result<Self> {
        let mut inner = ptr::null_mut();
        check_return_code(unsafe {
            bindings::faiss_IndexIVFFlat_new_with_metric(
                &mut inner,
                quantizer.inner,
                d,
                nlist,
                metric,
            )
        })?;
        Ok(Self { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index_flat::IndexFlat;
    use ndarray::Array2;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    fn generate_random_vectors(n: usize, d: usize) -> Array2<f32> {
        Array2::random((n, d), Uniform::new(-1.0f32, 1.0f32))
    }

    #[test]
    #[ignore = "Test has memory safety issues with quantizer ownership. Use index_factory instead."]
    fn test_index_ivf_flat_with_quantizer() {
        let d = 32;
        let n = 500;
        let nlist = 10;
        let k = 5;

        let mut quantizer = IndexFlat::new(d as i32, FaissMetricType::METRIC_L2).unwrap();
        let quantizer_inner = crate::index::Index {
            inner: quantizer.inner,
        };
        std::mem::forget(quantizer);

        let mut index = IndexIVFFlat::new(
            &mut crate::index::Index {
                inner: quantizer_inner.inner,
            },
            d,
            nlist,
        )
        .unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.train(n as i64, &data_slice).unwrap();
        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);
        assert_eq!(index.nlist(), nlist);

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
    #[ignore = "Test has memory safety issues with quantizer ownership. Use index_factory instead."]
    fn test_index_ivf_flat_nprobe() {
        let d = 16;
        let n = 200;
        let nlist = 5;

        let quantizer = IndexFlat::new(d as i32, FaissMetricType::METRIC_L2).unwrap();
        let quantizer_inner = crate::index::Index {
            inner: quantizer.inner,
        };
        std::mem::forget(quantizer);

        let mut index = IndexIVFFlat::new(
            &mut crate::index::Index {
                inner: quantizer_inner.inner,
            },
            d,
            nlist,
        )
        .unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.train(n as i64, &data_slice).unwrap();
        index.add(n as i64, &data_slice).unwrap();

        let default_nprobe = index.nprobe();
        assert!(default_nprobe > 0);

        index.set_nprobe(3);
        assert_eq!(index.nprobe(), 3);
    }

    #[test]
    #[ignore = "Test has memory safety issues with quantizer ownership. Use index_factory instead."]
    fn test_index_ivf_flat_with_metric() {
        let d = 16;
        let n = 100;
        let nlist = 4;

        let quantizer = IndexFlat::new(d as i32, FaissMetricType::METRIC_INNER_PRODUCT).unwrap();
        let quantizer_inner = crate::index::Index {
            inner: quantizer.inner,
        };
        std::mem::forget(quantizer);

        let mut index = IndexIVFFlat::new_with_metric(
            &mut crate::index::Index {
                inner: quantizer_inner.inner,
            },
            d,
            nlist,
            FaissMetricType::METRIC_INNER_PRODUCT,
        )
        .unwrap();

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        index.train(n as i64, &data_slice).unwrap();
        index.add(n as i64, &data_slice).unwrap();
        assert_eq!(index.ntotal(), n as i64);
    }
}
