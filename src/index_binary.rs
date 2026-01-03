use crate::error::FaissError;
use crate::ffi;
use crate::id_selector::IDSelectorTrait;
use crate::index::MetricType;
use crate::range_search::RangeSearchResult;
use anyhow::Result;

/// An owned wrapper around a FaissIndexBinary pointer
///
/// This struct takes ownership of a FaissIndexBinary pointer and ensures
/// it is properly freed when dropped. Binary indices work with binary (0/1)
/// vectors and typically use Hamming distance.
pub struct IndexBinaryOwned {
    inner: *mut ffi::FaissIndexBinary,
}

impl IndexBinaryOwned {
    /// Create a new IndexBinaryOwned from a raw FaissIndexBinary pointer
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The pointer is valid and points to a properly initialized FaissIndexBinary
    /// - The pointer is not used elsewhere after being passed to this function
    /// - The pointer was allocated by FAISS and can be freed with faiss_IndexBinary_free
    pub unsafe fn from_raw(ptr: *mut ffi::FaissIndexBinary) -> Result<Self> {
        if ptr.is_null() {
            anyhow::bail!("Cannot create IndexBinaryOwned from null pointer");
        }
        Ok(Self { inner: ptr })
    }

    /// Get a reference to the inner raw pointer
    pub fn as_ptr(&self) -> *mut ffi::FaissIndexBinary {
        self.inner
    }
}

impl Drop for IndexBinaryOwned {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IndexBinary_free(self.inner);
            }
        }
    }
}

impl IndexBinaryTrait for IndexBinaryOwned {
    fn inner_ptr(&self) -> *mut ffi::FaissIndexBinary {
        self.inner
    }
}

// Binary indices are generally not thread-safe for mutation
// Users can wrap in Arc<Mutex<>> if needed

/// A trait representing common operations for FAISS binary indices
///
/// Binary indices work with binary vectors (0/1 values) and typically use
/// Hamming distance for similarity measurement.
pub trait IndexBinaryTrait {
    /// Get the underlying raw pointer to the FAISS binary index
    fn inner_ptr(&self) -> *mut ffi::FaissIndexBinary;

    /// Get the dimension of the binary vectors in the index (in bits)
    fn d(&self) -> Result<i32> {
        unsafe {
            let d = ffi::faiss_IndexBinary_d(self.inner_ptr());
            Ok(d)
        }
    }

    /// Get the number of vectors in the index
    fn ntotal(&self) -> Result<i64> {
        unsafe {
            let n = ffi::faiss_IndexBinary_ntotal(self.inner_ptr());
            Ok(n)
        }
    }

    /// Check if the index is trained
    fn is_trained(&self) -> Result<bool> {
        unsafe {
            let trained = ffi::faiss_IndexBinary_is_trained(self.inner_ptr());
            Ok(trained != 0)
        }
    }

    /// Get the metric type used by this index
    fn metric_type(&self) -> Result<MetricType> {
        unsafe {
            let code = ffi::faiss_IndexBinary_metric_type(self.inner_ptr());
            MetricType::from_code(code)
                .ok_or_else(|| anyhow::anyhow!("Unknown metric type: {}", code))
        }
    }

    /// Get the verbosity level
    fn verbose(&self) -> Result<i32> {
        unsafe {
            let verbose = ffi::faiss_IndexBinary_verbose(self.inner_ptr());
            Ok(verbose)
        }
    }

    /// Set the verbosity level
    fn set_verbose(&mut self, level: i32) -> Result<()> {
        unsafe {
            ffi::faiss_IndexBinary_set_verbose(self.inner_ptr(), level);
        }
        Ok(())
    }

    /// Train the index on a set of binary vectors
    ///
    /// # Arguments
    /// * `n` - Number of training vectors
    /// * `x` - Training vectors (binary data, n * d/8 bytes)
    ///
    /// # Notes
    /// Binary vectors are packed into bytes (8 bits per byte).
    /// The input size should be n * (d / 8) bytes.
    fn train(&mut self, n: i64, x: &[u8]) -> Result<()> {
        let d = self.d()?;
        let expected_len = (n * ((d + 7) / 8) as i64) as usize;

        if x.len() != expected_len {
            anyhow::bail!(
                "Expected {} bytes (n={} * d={}/8), got {}",
                expected_len,
                n,
                d,
                x.len()
            );
        }

        unsafe {
            let ret = ffi::faiss_IndexBinary_train(self.inner_ptr(), n, x.as_ptr());
            if ret != 0 {
                anyhow::bail!("Failed to train binary index");
            }
        }
        Ok(())
    }

    /// Add binary vectors to the index
    ///
    /// # Arguments
    /// * `n` - Number of vectors to add
    /// * `x` - Binary vectors to add (n * d/8 bytes)
    fn add(&mut self, n: i64, x: &[u8]) -> Result<()> {
        let d = self.d()?;
        let expected_len = (n * ((d + 7) / 8) as i64) as usize;

        if x.len() != expected_len {
            anyhow::bail!(
                "Expected {} bytes (n={} * d={}/8), got {}",
                expected_len,
                n,
                d,
                x.len()
            );
        }

        unsafe {
            let ret = ffi::faiss_IndexBinary_add(self.inner_ptr(), n, x.as_ptr());
            if ret != 0 {
                anyhow::bail!("Failed to add vectors to binary index");
            }
        }
        Ok(())
    }

    /// Add binary vectors with explicit IDs to the index
    ///
    /// # Arguments
    /// * `n` - Number of vectors to add
    /// * `x` - Binary vectors to add (n * d/8 bytes)
    /// * `xids` - IDs for the vectors (n values)
    fn add_with_ids(&mut self, n: i64, x: &[u8], xids: &[i64]) -> Result<()> {
        let d = self.d()?;
        let expected_len = (n * ((d + 7) / 8) as i64) as usize;

        if x.len() != expected_len {
            anyhow::bail!(
                "Expected {} bytes (n={} * d={}/8), got {}",
                expected_len,
                n,
                d,
                x.len()
            );
        }

        if xids.len() != n as usize {
            anyhow::bail!("Expected {} IDs, got {}", n, xids.len());
        }

        unsafe {
            let ret = ffi::faiss_IndexBinary_add_with_ids(
                self.inner_ptr(),
                n,
                x.as_ptr(),
                xids.as_ptr(),
            );
            if ret != 0 {
                anyhow::bail!("Failed to add vectors with IDs to binary index");
            }
        }
        Ok(())
    }

    /// Search the binary index for nearest neighbors
    ///
    /// # Arguments
    /// * `n` - Number of query vectors
    /// * `x` - Query binary vectors (n * d/8 bytes)
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// A tuple of (distances, labels) where each is a vector of length n*k
    /// Distances are Hamming distances (integer values)
    fn search(&self, n: i64, x: &[u8], k: i64) -> Result<(Vec<i32>, Vec<i64>)> {
        let d = self.d()?;
        let expected_len = (n * ((d + 7) / 8) as i64) as usize;

        if x.len() != expected_len {
            anyhow::bail!(
                "Expected {} bytes (n={} * d={}/8), got {}",
                expected_len,
                n,
                d,
                x.len()
            );
        }

        let mut distances = vec![0i32; (n * k) as usize];
        let mut labels = vec![0i64; (n * k) as usize];

        unsafe {
            let ret = ffi::faiss_IndexBinary_search(
                self.inner_ptr(),
                n,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
            );
            if ret != 0 {
                anyhow::bail!("Failed to search binary index");
            }
        }

        Ok((distances, labels))
    }

    /// Range search: find all vectors within a given Hamming radius
    ///
    /// # Arguments
    /// * `n` - Number of query vectors
    /// * `x` - Query binary vectors (n * d/8 bytes)
    /// * `radius` - Search radius (Hamming distance threshold)
    /// * `result` - RangeSearchResult to store the results
    fn range_search(&self, n: i64, x: &[u8], radius: i32, result: &mut RangeSearchResult) -> Result<()> {
        let d = self.d()?;
        let expected_len = (n * ((d + 7) / 8) as i64) as usize;

        if x.len() != expected_len {
            anyhow::bail!(
                "Expected {} bytes (n={} * d={}/8), got {}",
                expected_len,
                n,
                d,
                x.len()
            );
        }

        unsafe {
            let ret = ffi::faiss_IndexBinary_range_search(
                self.inner_ptr(),
                n,
                x.as_ptr(),
                radius,
                result.as_ptr(),
            );
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
        }

        Ok(())
    }

    /// Assign query vectors to their nearest cluster
    ///
    /// # Arguments
    /// * `n` - Number of query vectors
    /// * `x` - Query binary vectors (n * d/8 bytes)
    /// * `k` - Number of nearest clusters to assign
    ///
    /// # Returns
    /// A vector of cluster assignments (length n * k)
    fn assign(&self, n: i64, x: &[u8], k: i64) -> Result<Vec<i64>> {
        let d = self.d()?;
        let expected_len = (n * ((d + 7) / 8) as i64) as usize;

        if x.len() != expected_len {
            anyhow::bail!(
                "Expected {} bytes (n={} * d={}/8), got {}",
                expected_len,
                n,
                d,
                x.len()
            );
        }

        let mut labels = vec![0i64; (n * k) as usize];

        unsafe {
            let ret = ffi::faiss_IndexBinary_assign(
                self.inner_ptr(),
                n,
                x.as_ptr(),
                labels.as_mut_ptr(),
                k,
            );
            if ret != 0 {
                anyhow::bail!("Failed to assign vectors");
            }
        }

        Ok(labels)
    }

    /// Reset the index, removing all vectors
    fn reset(&mut self) -> Result<()> {
        unsafe {
            let ret = ffi::faiss_IndexBinary_reset(self.inner_ptr());
            if ret != 0 {
                anyhow::bail!("Failed to reset binary index");
            }
        }
        Ok(())
    }

    /// Remove vectors from the index using an ID selector
    ///
    /// # Arguments
    /// * `selector` - The ID selector that determines which IDs to remove
    ///
    /// # Returns
    /// The number of vectors actually removed
    fn remove_ids(&mut self, selector: &dyn IDSelectorTrait) -> Result<usize> {
        unsafe {
            let mut n_removed: usize = 0;
            let ret = ffi::faiss_IndexBinary_remove_ids(
                self.inner_ptr(),
                selector.as_ptr(),
                &mut n_removed,
            );

            if ret != 0 {
                anyhow::bail!("Failed to remove IDs from binary index");
            }

            Ok(n_removed)
        }
    }

    /// Reconstruct a vector from the index
    ///
    /// # Arguments
    /// * `key` - The ID of the vector to reconstruct
    ///
    /// # Returns
    /// The reconstructed binary vector (d/8 bytes)
    fn reconstruct(&self, key: i64) -> Result<Vec<u8>> {
        let d = self.d()?;
        let vec_size = ((d + 7) / 8) as usize;
        let mut recons = vec![0u8; vec_size];

        unsafe {
            let ret = ffi::faiss_IndexBinary_reconstruct(
                self.inner_ptr(),
                key,
                recons.as_mut_ptr(),
            );
            if ret != 0 {
                anyhow::bail!("Failed to reconstruct vector with key {}", key);
            }
        }

        Ok(recons)
    }

    /// Reconstruct multiple vectors from the index
    ///
    /// # Arguments
    /// * `n` - Number of vectors to reconstruct
    /// * `keys` - The IDs of the vectors to reconstruct
    ///
    /// # Returns
    /// The reconstructed binary vectors (n * d/8 bytes)
    fn reconstruct_n(&self, n: i64, keys: &[i64]) -> Result<Vec<u8>> {
        if keys.len() != n as usize {
            anyhow::bail!("Expected {} keys, got {}", n, keys.len());
        }

        let d = self.d()?;
        let vec_size = ((d + 7) / 8) as usize;
        let mut recons = vec![0u8; n as usize * vec_size];

        unsafe {
            let ret = ffi::faiss_IndexBinary_reconstruct_n(
                self.inner_ptr(),
                0,
                n,
                recons.as_mut_ptr(),
            );
            if ret != 0 {
                anyhow::bail!("Failed to reconstruct {} vectors", n);
            }
        }

        Ok(recons)
    }
}

/// Binary IVF (Inverted File) index wrapper
///
/// Provides access to IVF-specific parameters and methods for binary indices.
pub struct IndexBinaryIVF {
    inner: *mut ffi::FaissIndexBinaryIVF,
}

impl IndexBinaryIVF {
    /// Cast a binary index to IndexBinaryIVF
    ///
    /// # Safety
    /// The caller must ensure the index is actually an IVF variant
    pub unsafe fn from_index(index: &IndexBinaryOwned) -> Result<Self> {
        unsafe {
            let ivf_ptr = ffi::faiss_IndexBinaryIVF_cast(index.as_ptr());
            if ivf_ptr.is_null() {
                anyhow::bail!("Index is not an IndexBinaryIVF variant");
            }
            Ok(Self { inner: ivf_ptr })
        }
    }

    /// Get the number of inverted lists (clusters)
    pub fn nlist(&self) -> usize {
        unsafe { ffi::faiss_IndexBinaryIVF_nlist(self.inner) }
    }

    /// Get the number of lists to probe during search
    pub fn nprobe(&self) -> usize {
        unsafe { ffi::faiss_IndexBinaryIVF_nprobe(self.inner) }
    }

    /// Set the number of lists to probe during search
    pub fn set_nprobe(&mut self, nprobe: usize) {
        unsafe {
            ffi::faiss_IndexBinaryIVF_set_nprobe(self.inner, nprobe);
        }
    }

    /// Get the quantizer index
    pub fn quantizer(&self) -> Result<*mut ffi::FaissIndexBinary> {
        unsafe {
            let quantizer_ptr = ffi::faiss_IndexBinaryIVF_quantizer(self.inner);
            if quantizer_ptr.is_null() {
                anyhow::bail!("Failed to get quantizer");
            }
            Ok(quantizer_ptr)
        }
    }

    /// Check if the index owns its internal fields
    pub fn own_fields(&self) -> bool {
        unsafe {
            let owns = ffi::faiss_IndexBinaryIVF_own_fields(self.inner);
            owns != 0
        }
    }

    /// Set whether the index owns its internal fields
    pub fn set_own_fields(&mut self, own: bool) -> Result<()> {
        unsafe {
            ffi::faiss_IndexBinaryIVF_set_own_fields(self.inner, own as i32);
        }
        Ok(())
    }

    /// Get the maximum number of codes per inverted list
    pub fn max_codes(&self) -> usize {
        unsafe { ffi::faiss_IndexBinaryIVF_max_codes(self.inner) }
    }

    /// Set the maximum number of codes per inverted list
    pub fn set_max_codes(&mut self, max_codes: usize) {
        unsafe {
            ffi::faiss_IndexBinaryIVF_set_max_codes(self.inner, max_codes);
        }
    }

    /// Check if heap-based search is used
    pub fn use_heap(&self) -> bool {
        unsafe { ffi::faiss_IndexBinaryIVF_use_heap(self.inner) != 0 }
    }

    /// Set whether to use heap-based search
    pub fn set_use_heap(&mut self, use_heap: bool) {
        unsafe {
            ffi::faiss_IndexBinaryIVF_set_use_heap(self.inner, use_heap as i32);
        }
    }

    /// Get the imbalance factor of the inverted lists
    ///
    /// Returns the ratio between the largest and average list size
    pub fn imbalance_factor(&self) -> f64 {
        unsafe { ffi::faiss_IndexBinaryIVF_imbalance_factor(self.inner) }
    }

    /// Print statistics about the inverted lists
    pub fn print_stats(&self) {
        unsafe {
            ffi::faiss_IndexBinaryIVF_print_stats(self.inner);
        }
    }
}
