use crate::error::FaissError;
use crate::ffi;
use crate::index::{IndexOwned, IndexTrait, MetricType};
use crate::index_ivf::IndexIVFTrait;
use anyhow::Result;

/// IVF index with flat (uncompressed) storage
///
/// IndexIVFFlat is an IVF index where vectors are stored without compression.
/// It provides exact distances within the searched lists.
///
/// # Characteristics
/// - Exact distances (no quantization error within searched lists)
/// - Memory usage: O(ntotal * d) floats
/// - Good baseline for accuracy vs. speed trade-off
/// - Faster than flat index, more accurate than compressed IVF variants
///
/// # When to Use
/// - Medium-sized datasets (100K-10M vectors)
/// - When you need exact distances within clusters
/// - As a baseline before trying compressed variants (IVF-PQ, etc.)
///
/// # Performance
/// - Training: O(nlist * d * training_size)
/// - Search: O(nprobe * vectors_per_list * d)
/// - Memory: Same as flat index but with inverted list overhead
pub struct IndexIVFFlat {
    inner: *mut ffi::FaissIndexIVFFlat,
}

impl IndexIVFFlat {
    /// Create a new IVF flat index (requires manual initialization)
    ///
    /// # Note
    /// You probably want to use `new_with` or `new_with_metric` instead.
    pub fn new() -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexIVFFlat_new(&mut index_ptr);

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Create a new IVF flat index with quantizer
    ///
    /// # Arguments
    /// * `quantizer` - The quantizer index (usually IndexFlatL2 or IndexFlatIP)
    /// * `d` - Vector dimension
    /// * `nlist` - Number of inverted lists (clusters)
    ///
    /// # Example
    /// ```ignore
    /// use faiss_next::{IndexFlatL2, IndexIVFFlat};
    ///
    /// let quantizer = IndexFlatL2::new(128)?;
    /// let index = IndexIVFFlat::new_with(&quantizer, 128, 100)?;
    /// ```
    ///
    /// # Choosing nlist
    /// - Small datasets (<100K): 100-1000
    /// - Medium datasets (100K-1M): 1000-10000
    /// - Large datasets (>1M): 10000-100000
    /// - Rule of thumb: sqrt(n) to 4*sqrt(n)
    pub fn new_with(quantizer: &impl IndexTrait, d: usize, nlist: usize) -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexIVFFlat_new_with(
                &mut index_ptr,
                quantizer.inner_ptr(),
                d,
                nlist,
            );

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Create a new IVF flat index with specified metric
    ///
    /// # Arguments
    /// * `quantizer` - The quantizer index
    /// * `d` - Vector dimension
    /// * `nlist` - Number of inverted lists
    /// * `metric` - Distance metric (L2, InnerProduct, etc.)
    ///
    /// # Example
    /// ```ignore
    /// use faiss_next::{IndexFlatIP, IndexIVFFlat, MetricType};
    ///
    /// let quantizer = IndexFlatIP::new(128)?;
    /// let index = IndexIVFFlat::new_with_metric(
    ///     &quantizer,
    ///     128,
    ///     100,
    ///     MetricType::InnerProduct
    /// )?;
    /// ```
    pub fn new_with_metric(
        quantizer: &impl IndexTrait,
        d: usize,
        nlist: usize,
        metric: MetricType,
    ) -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexIVFFlat_new_with_metric(
                &mut index_ptr,
                quantizer.inner_ptr(),
                d,
                nlist,
                metric.to_code(),
            );

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Cast a generic index to IndexIVFFlat
    ///
    /// # Arguments
    /// * `index` - The index to cast
    ///
    /// # Returns
    /// `Some(IndexIVFFlat)` if the index is actually an IVFFlat index, `None` otherwise
    pub fn from_index(index: &IndexOwned) -> Option<Self> {
        unsafe {
            let ivf_ptr = ffi::faiss_IndexIVFFlat_cast(index.as_ptr());
            if ivf_ptr.is_null() {
                None
            } else {
                Some(Self { inner: ivf_ptr })
            }
        }
    }

    /// Convert to IndexOwned
    ///
    /// Consumes this index and returns it as a generic IndexOwned.
    pub fn into_index(self) -> Result<IndexOwned> {
        let ptr = self.inner;
        std::mem::forget(self);
        unsafe { IndexOwned::from_raw(ptr as *mut ffi::FaissIndex) }
    }

    /// Add vectors with explicit assignment to clusters
    ///
    /// This is a low-level function that adds vectors with pre-computed
    /// cluster assignments.
    ///
    /// # Arguments
    /// * `n` - Number of vectors to add
    /// * `x` - Vector data (n * d floats)
    /// * `xids` - Vector IDs (n values), can be None for sequential IDs
    /// * `precomputed_idx` - Pre-computed cluster assignments (n values), can be None
    ///
    /// # Example
    /// ```ignore
    /// // Add vectors with automatic cluster assignment
    /// index.add_core(n, &vectors, None, None)?;
    ///
    /// // Add vectors with specific IDs and pre-computed assignments
    /// index.add_core(n, &vectors, Some(&ids), Some(&assignments))?;
    /// ```
    pub fn add_core(
        &mut self,
        n: i64,
        x: &[f32],
        xids: Option<&[i64]>,
        precomputed_idx: Option<&[i64]>,
    ) -> Result<()> {
        let d = self.d()?;
        let expected_len = (n * d as i64) as usize;

        if x.len() != expected_len {
            anyhow::bail!(
                "Expected {} values (n={} * d={}), got {}",
                expected_len,
                n,
                d,
                x.len()
            );
        }

        if let Some(ids) = xids
            && ids.len() != n as usize
        {
            anyhow::bail!("Expected {} IDs, got {}", n, ids.len());
        }

        if let Some(idx) = precomputed_idx
            && idx.len() != n as usize
        {
            anyhow::bail!("Expected {} precomputed indices, got {}", n, idx.len());
        }

        unsafe {
            let xids_ptr = xids.map_or(std::ptr::null(), |ids| ids.as_ptr());
            let idx_ptr = precomputed_idx.map_or(std::ptr::null(), |idx| idx.as_ptr());

            let ret = ffi::faiss_IndexIVFFlat_add_core(
                self.inner,
                n,
                x.as_ptr(),
                xids_ptr,
                idx_ptr,
            );

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
        }

        Ok(())
    }

    /// Update vectors in the index
    ///
    /// Updates the values of existing vectors. The index must have a direct map
    /// enabled (see `make_direct_map` in IndexIVFTrait).
    ///
    /// # Arguments
    /// * `idx` - Vector IDs to update
    /// * `v` - New vector values (idx.len() * d floats)
    ///
    /// # Example
    /// ```ignore
    /// // Enable direct map for fast updates
    /// index.make_direct_map(true)?;
    ///
    /// // Update vectors 0 and 5
    /// let ids = vec![0, 5];
    /// let new_values = vec![/* 2 * d floats */];
    /// index.update_vectors(&ids, &new_values)?;
    /// ```
    ///
    /// # Errors
    /// Returns an error if:
    /// - The direct map is not enabled
    /// - Any ID is not found in the index
    /// - The vector data size doesn't match
    pub fn update_vectors(&mut self, idx: &mut [i64], v: &[f32]) -> Result<()> {
        let d = self.d()?;
        let nv = idx.len();
        let expected_len = nv * d as usize;

        if v.len() != expected_len {
            anyhow::bail!(
                "Expected {} values ({} vectors * d={}), got {}",
                expected_len,
                nv,
                d,
                v.len()
            );
        }

        unsafe {
            let ret = ffi::faiss_IndexIVFFlat_update_vectors(
                self.inner,
                nv as i32,
                idx.as_mut_ptr(),
                v.as_ptr(),
            );

            if ret != 0 {
                anyhow::bail!("Failed to update vectors (is direct map enabled?)");
            }
        }

        Ok(())
    }

    /// Get the inner pointer
    pub fn as_ptr(&self) -> *mut ffi::FaissIndexIVFFlat {
        self.inner
    }
}

impl Drop for IndexIVFFlat {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IndexIVFFlat_free(self.inner);
            }
        }
    }
}

impl IndexTrait for IndexIVFFlat {
    fn inner_ptr(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut ffi::FaissIndex
    }
}

impl IndexIVFTrait for IndexIVFFlat {
    fn inner_ivf_ptr(&self) -> *mut ffi::FaissIndexIVF {
        self.inner as *mut ffi::FaissIndexIVF
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index_flat::IndexFlatL2;

    #[test]
    fn test_index_ivf_flat_creation() {
        let quantizer = IndexFlatL2::new(4).unwrap();
        let index = IndexIVFFlat::new_with(&quantizer, 4, 10);
        assert!(index.is_ok(), "Failed to create IndexIVFFlat");

        let index = index.unwrap();
        assert_eq!(index.d().unwrap(), 4);
        assert_eq!(index.nlist(), 10);
        assert!(!index.is_trained().unwrap());
    }

    #[test]
    fn test_index_ivf_flat_train_and_add() {
        let quantizer = IndexFlatL2::new(4).unwrap();
        let mut index = IndexIVFFlat::new_with(&quantizer, 4, 5).unwrap();

        // Train the index
        let training_vectors = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 0.0, 1.0,
        ];
        index.train(8, &training_vectors).unwrap();
        assert!(index.is_trained().unwrap());

        // Add vectors
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        index.add(2, &vectors).unwrap();
        assert_eq!(index.ntotal().unwrap(), 2);
    }

    #[test]
    fn test_index_ivf_flat_search() {
        let quantizer = IndexFlatL2::new(4).unwrap();
        let mut index = IndexIVFFlat::new_with(&quantizer, 4, 3).unwrap();

        // Train
        let training_vectors = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
        ];
        index.train(6, &training_vectors).unwrap();

        // Add vectors
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        index.add(3, &vectors).unwrap();

        // Search
        index.set_nprobe(3);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let (distances, labels) = index.search(1, &query, 1).unwrap();

        assert_eq!(labels[0], 0);
        assert!(distances[0] < 0.001);
    }

    #[test]
    fn test_index_ivf_flat_add_core_validation() {
        let quantizer = IndexFlatL2::new(4).unwrap();
        let mut index = IndexIVFFlat::new_with(&quantizer, 4, 3).unwrap();

        // Train
        let training_vectors = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0,
        ];
        index.train(5, &training_vectors).unwrap();

        // Test that add_core validates input sizes correctly
        let vectors = vec![1.0, 0.0]; // Only 2 elements, should fail
        let result = index.add_core(2, &vectors, None, None);
        assert!(result.is_err(), "Should fail with wrong vector size");

        // Test with correct vector size but wrong ID count
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        let ids = vec![100]; // Only 1 ID for 2 vectors
        let result = index.add_core(2, &vectors, Some(&ids), None);
        assert!(result.is_err(), "Should fail with wrong ID count");
    }

    #[test]
    fn test_index_ivf_flat_update_vectors() {
        let quantizer = IndexFlatL2::new(4).unwrap();
        let mut index = IndexIVFFlat::new_with(&quantizer, 4, 3).unwrap();

        // Train
        let training_vectors = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
        ];
        index.train(4, &training_vectors).unwrap();

        // Add vectors
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        index.add(2, &vectors).unwrap();

        // Enable direct map for updates
        index.make_direct_map(true).unwrap();

        // Update vector 0
        let mut ids = vec![0];
        let new_values = vec![2.0, 0.0, 0.0, 0.0];
        index.update_vectors(&mut ids, &new_values).unwrap();

        // Verify update by searching
        index.set_nprobe(3);
        let query = vec![2.0, 0.0, 0.0, 0.0];
        let (distances, labels) = index.search(1, &query, 1).unwrap();

        assert_eq!(labels[0], 0);
        assert!(distances[0] < 0.001);
    }

    #[test]
    fn test_index_ivf_flat_with_metric() {
        let quantizer = IndexFlatL2::new(4).unwrap();
        let index = IndexIVFFlat::new_with_metric(
            &quantizer,
            4,
            5,
            MetricType::L2,
        );
        assert!(index.is_ok());

        let index = index.unwrap();
        assert_eq!(index.metric_type().unwrap(), MetricType::L2);
    }
}
