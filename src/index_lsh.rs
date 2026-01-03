use crate::error::FaissError;
use crate::ffi;
use crate::index::{IndexOwned, IndexTrait};
use anyhow::Result;

/// Locality Sensitive Hashing (LSH) index
///
/// IndexLSH uses random projections to create binary signatures for vectors.
/// The sign of each vector component after projection is stored as a bit in
/// the signature, creating a compact binary representation.
///
/// # How it Works
/// 1. **Training**: Generate random rotation matrix (if rotate_data is true)
/// 2. **Adding**: Project vectors and store their binary signatures
/// 3. **Searching**: Find vectors with similar binary signatures using Hamming distance
///
/// # Characteristics
/// - Very fast search (Hamming distance on binary codes)
/// - Compact storage (nbits bits per vector)
/// - Approximate search only
/// - Best for high-dimensional data
///
/// # When to Use
/// - Need very fast approximate search
/// - Memory is constrained
/// - Working with high-dimensional vectors (>100D)
/// - Can tolerate lower recall
///
/// # Performance
/// - Training: O(d * nbits) for rotation matrix
/// - Search: O(nbits) for Hamming distance
/// - Memory: O(ntotal * nbits / 8) bytes
///
/// # Example
/// ```ignore
/// use faiss_next::IndexLSH;
///
/// // Create LSH index with 128 bits
/// let mut index = IndexLSH::new(128, 128)?;
///
/// // Train and add vectors
/// index.train(1000, &training_data)?;
/// index.add(1000, &vectors)?;
///
/// // Search
/// let (distances, labels) = index.search(1, &query, 10)?;
/// ```
pub struct IndexLSH {
    inner: *mut ffi::FaissIndexLSH,
}

impl IndexLSH {
    /// Create a new LSH index
    ///
    /// # Arguments
    /// * `d` - Vector dimension
    /// * `nbits` - Number of bits in the hash (code size will be nbits/8 bytes)
    ///
    /// # Example
    /// ```ignore
    /// let index = IndexLSH::new(128, 256)?;
    /// ```
    ///
    /// # Choosing nbits
    /// - Larger nbits = more accurate but slower search and more memory
    /// - Typical values: d to 4*d
    /// - Must be multiple of 8 for efficient storage
    pub fn new(d: i64, nbits: i32) -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexLSH_new(&mut index_ptr, d, nbits);

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Create a new LSH index with options
    ///
    /// # Arguments
    /// * `d` - Vector dimension
    /// * `nbits` - Number of bits in the hash
    /// * `rotate_data` - If true, apply random rotation before hashing
    /// * `train_thresholds` - If true, train thresholds instead of using 0
    ///
    /// # Example
    /// ```ignore
    /// // Create index with rotation and trained thresholds
    /// let index = IndexLSH::new_with_options(128, 256, true, true)?;
    /// ```
    ///
    /// # Options
    /// - `rotate_data=true`: More uniform bit distribution, better quality
    /// - `train_thresholds=true`: Learn optimal thresholds from training data
    pub fn new_with_options(
        d: i64,
        nbits: i32,
        rotate_data: bool,
        train_thresholds: bool,
    ) -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexLSH_new_with_options(
                &mut index_ptr,
                d,
                nbits,
                rotate_data as i32,
                train_thresholds as i32,
            );

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Cast a generic index to IndexLSH
    ///
    /// # Arguments
    /// * `index` - The index to cast
    ///
    /// # Returns
    /// `Some(IndexLSH)` if the index is actually an LSH index, `None` otherwise
    pub fn from_index(index: &IndexOwned) -> Option<Self> {
        unsafe {
            let lsh_ptr = ffi::faiss_IndexLSH_cast(index.as_ptr());
            if lsh_ptr.is_null() {
                None
            } else {
                Some(Self { inner: lsh_ptr })
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

    /// Get the number of bits in the hash
    ///
    /// This is the size of the binary signature for each vector.
    pub fn nbits(&self) -> i32 {
        unsafe { ffi::faiss_IndexLSH_nbits(self.inner) }
    }

    /// Get the code size in bytes
    ///
    /// This is nbits / 8 (rounded up).
    pub fn code_size(&self) -> i32 {
        unsafe { ffi::faiss_IndexLSH_code_size(self.inner) }
    }

    /// Check if data rotation is enabled
    ///
    /// Returns true if vectors are rotated before hashing.
    pub fn rotate_data(&self) -> bool {
        unsafe { ffi::faiss_IndexLSH_rotate_data(self.inner) != 0 }
    }

    /// Check if threshold training is enabled
    ///
    /// Returns true if thresholds are learned during training.
    pub fn train_thresholds(&self) -> bool {
        unsafe { ffi::faiss_IndexLSH_train_thresholds(self.inner) != 0 }
    }

    /// Get the inner pointer
    pub fn as_ptr(&self) -> *mut ffi::FaissIndexLSH {
        self.inner
    }
}

impl Drop for IndexLSH {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IndexLSH_free(self.inner);
            }
        }
    }
}

impl IndexTrait for IndexLSH {
    fn inner_ptr(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut ffi::FaissIndex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_lsh_creation() {
        let index = IndexLSH::new(128, 128);
        assert!(index.is_ok(), "Failed to create IndexLSH");

        let index = index.unwrap();
        assert_eq!(index.d().unwrap(), 128);
        assert_eq!(index.nbits(), 128);
        assert_eq!(index.code_size(), 16); // 128 bits = 16 bytes
        // LSH doesn't need training if train_thresholds is false
    }

    #[test]
    fn test_index_lsh_with_options() {
        let index = IndexLSH::new_with_options(64, 32, true, true);
        assert!(index.is_ok());

        let index = index.unwrap();
        assert_eq!(index.nbits(), 32);
        assert!(index.rotate_data());
        assert!(index.train_thresholds());
    }

    #[test]
    fn test_index_lsh_no_rotation() {
        let index = IndexLSH::new_with_options(32, 16, false, false).unwrap();
        assert!(!index.rotate_data());
        assert!(!index.train_thresholds());
    }

    #[test]
    fn test_index_lsh_train_and_add() {
        let mut index = IndexLSH::new_with_options(4, 4, false, true).unwrap();

        // Train the index (needed when train_thresholds is true)
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
    fn test_index_lsh_search() {
        let mut index = IndexLSH::new_with_options(64, 32, true, false).unwrap();

        // Train (not strictly needed when train_thresholds=false, but doesn't hurt)
        let mut training_vectors = Vec::new();
        for i in 0..6 {
            for j in 0..64 {
                training_vectors.push(if j == i % 64 { 1.0 } else { 0.0 });
            }
        }
        index.train(6, &training_vectors).unwrap();

        // Add vectors
        let mut vectors = Vec::new();
        for i in 0..3 {
            for j in 0..64 {
                vectors.push(if j == i { 1.0 } else { 0.0 });
            }
        }
        index.add(3, &vectors).unwrap();

        // Search
        let mut query = vec![0.0; 64];
        query[0] = 1.0;
        let (distances, labels) = index.search(1, &query, 1).unwrap();

        // LSH is approximate, so we just check that search works
        assert_eq!(labels.len(), 1);
        assert_eq!(distances.len(), 1);
    }

    #[test]
    fn test_index_lsh_code_size() {
        // Test various bit sizes (d must be >= nbits)
        let index8 = IndexLSH::new(16, 8).unwrap();
        assert_eq!(index8.code_size(), 1);

        let index16 = IndexLSH::new(32, 16).unwrap();
        assert_eq!(index16.code_size(), 2);

        let index64 = IndexLSH::new(128, 64).unwrap();
        assert_eq!(index64.code_size(), 8);
    }
}
