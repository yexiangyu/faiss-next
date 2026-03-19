use crate::bindings;
use crate::error::{check_return_code, Result};
use crate::macros::*;
use crate::traits::{FaissIVFIndex, FaissIndex};

pub struct IndexIVF {
    pub(crate) inner: *mut bindings::FaissIndex,
}

impl_faiss_drop!(IndexIVF, faiss_IndexIVF_free);
impl_index_common!(IndexIVF);

impl FaissIndex for IndexIVF {
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

impl FaissIVFIndex for IndexIVF {
    fn nlist(&self) -> usize {
        unsafe { bindings::faiss_IndexIVF_nlist(self.inner as *mut _) }
    }

    fn nprobe(&self) -> usize {
        unsafe { bindings::faiss_IndexIVF_nprobe(self.inner as *mut _) }
    }

    fn set_nprobe(&mut self, nprobe: usize) {
        unsafe { bindings::faiss_IndexIVF_set_nprobe(self.inner as *mut _, nprobe) }
    }

    fn quantizer(&self) -> *mut bindings::FaissIndex {
        unsafe { bindings::faiss_IndexIVF_quantizer(self.inner as *mut _) }
    }
}

impl IndexIVF {
    pub fn from_index(index: crate::index::Index) -> Self {
        let inner = index.inner;
        std::mem::forget(index);
        Self { inner }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::FaissMetricType;
    use crate::index_factory::index_factory;
    use ndarray::Array2;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    fn generate_random_vectors(n: usize, d: usize) -> Array2<f32> {
        Array2::random((n, d), Uniform::new(-1.0f32, 1.0f32))
    }

    #[test]
    fn test_index_ivf_basic() {
        let d = 32;
        let n = 1000;
        let nlist = 10;
        let k = 5;

        let mut index = index_factory(
            d as i32,
            &format!("IVF{},Flat", nlist),
            FaissMetricType::METRIC_L2,
        )
        .unwrap();
        let mut ivf_index = IndexIVF::from_index(index);

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        ivf_index.train(n as i64, &data_slice).unwrap();
        ivf_index.add(n as i64, &data_slice).unwrap();
        assert_eq!(ivf_index.ntotal(), n as i64);
        assert_eq!(ivf_index.nlist(), nlist);

        let query = data.row(100).to_owned();
        let mut distances = vec![0.0f32; k];
        let mut labels = vec![-1i64; k];

        ivf_index
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
    fn test_index_ivf_nprobe() {
        let d = 16;
        let n = 500;
        let nlist = 5;

        let mut index = index_factory(
            d as i32,
            &format!("IVF{},Flat", nlist),
            FaissMetricType::METRIC_L2,
        )
        .unwrap();
        let mut ivf_index = IndexIVF::from_index(index);

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        ivf_index.train(n as i64, &data_slice).unwrap();
        ivf_index.add(n as i64, &data_slice).unwrap();

        let default_nprobe = ivf_index.nprobe();
        assert!(default_nprobe > 0);

        ivf_index.set_nprobe(3);
        assert_eq!(ivf_index.nprobe(), 3);
    }

    #[test]
    fn test_index_ivf_add_with_ids() {
        let d = 16;
        let n = 200;
        let nlist = 4;

        let mut index = index_factory(
            d as i32,
            &format!("IVF{},Flat", nlist),
            FaissMetricType::METRIC_L2,
        )
        .unwrap();
        let mut ivf_index = IndexIVF::from_index(index);

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();
        let ids: Vec<i64> = (1000..1000 + n as i64).collect();

        ivf_index.train(n as i64, &data_slice).unwrap();
        ivf_index.add_with_ids(n as i64, &data_slice, &ids).unwrap();
        assert_eq!(ivf_index.ntotal(), n as i64);

        let query = data.row(50).to_owned();
        let mut distances = vec![0.0f32; 1];
        let mut labels = vec![-1i64; 1];

        ivf_index
            .search(1, query.as_slice().unwrap(), 1, &mut distances, &mut labels)
            .unwrap();

        assert!(labels[0] >= 1000);
    }

    #[test]
    fn test_index_ivf_reset() {
        let d = 8;
        let n = 100;
        let nlist = 4;

        let mut index = index_factory(
            d as i32,
            &format!("IVF{},Flat", nlist),
            FaissMetricType::METRIC_L2,
        )
        .unwrap();
        let mut ivf_index = IndexIVF::from_index(index);

        let data = generate_random_vectors(n, d);
        let data_slice: Vec<f32> = data.iter().copied().collect();

        ivf_index.train(n as i64, &data_slice).unwrap();
        ivf_index.add(n as i64, &data_slice).unwrap();
        assert_eq!(ivf_index.ntotal(), n as i64);

        ivf_index.reset().unwrap();
        assert_eq!(ivf_index.ntotal(), 0);
    }
}
