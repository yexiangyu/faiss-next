use crate::error::FaissError;
use crate::ffi;
use anyhow::Result;

/// Result of a range search operation
///
/// Range search finds all vectors within a given radius of each query vector.
/// Results are stored in a CSR-like format where:
/// - `lims[i]` to `lims[i+1]` gives the range of results for query i
/// - `labels[lims[i]:lims[i+1]]` contains the IDs of matching vectors
/// - `distances[lims[i]:lims[i+1]]` contains the corresponding distances
pub struct RangeSearchResult {
    inner: *mut ffi::FaissRangeSearchResult,
}

impl RangeSearchResult {
    /// Create a new RangeSearchResult for a given number of queries
    ///
    /// # Arguments
    /// * `nq` - Number of queries
    pub fn new(nq: i64) -> Result<Self> {
        unsafe {
            let mut rsr = std::ptr::null_mut();
            let ret = ffi::faiss_RangeSearchResult_new(&mut rsr, nq);
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(Self { inner: rsr })
        }
    }

    /// Create a new RangeSearchResult with optional allocation of lims array
    ///
    /// # Arguments
    /// * `nq` - Number of queries
    /// * `alloc_lims` - Whether to allocate the lims array
    pub fn new_with(nq: i64, alloc_lims: bool) -> Result<Self> {
        unsafe {
            let mut rsr = std::ptr::null_mut();
            let ret = ffi::faiss_RangeSearchResult_new_with(&mut rsr, nq, alloc_lims as i32);
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(Self { inner: rsr })
        }
    }

    /// Get the number of queries
    pub fn nq(&self) -> usize {
        unsafe { ffi::faiss_RangeSearchResult_nq(self.inner) }
    }

    /// Get the buffer size (total number of results across all queries)
    pub fn buffer_size(&self) -> usize {
        unsafe { ffi::faiss_RangeSearchResult_buffer_size(self.inner) }
    }

    /// Get the lims array
    ///
    /// # Returns
    /// A slice of size nq+1 where lims[i] to lims[i+1] gives the range for query i
    pub fn lims(&self) -> Result<&[usize]> {
        unsafe {
            let mut lims_ptr: *mut usize = std::ptr::null_mut();
            ffi::faiss_RangeSearchResult_lims(self.inner, &mut lims_ptr);

            if lims_ptr.is_null() {
                anyhow::bail!("Failed to get lims from RangeSearchResult");
            }

            Ok(std::slice::from_raw_parts(lims_ptr, self.nq() + 1))
        }
    }

    /// Get the labels and distances arrays
    ///
    /// # Returns
    /// A tuple of (labels, distances) where each is a slice of size buffer_size()
    pub fn labels_and_distances(&self) -> Result<(&[i64], &[f32])> {
        unsafe {
            let mut labels_ptr: *mut i64 = std::ptr::null_mut();
            let mut distances_ptr: *mut f32 = std::ptr::null_mut();

            ffi::faiss_RangeSearchResult_labels(
                self.inner,
                &mut labels_ptr,
                &mut distances_ptr,
            );

            if labels_ptr.is_null() || distances_ptr.is_null() {
                anyhow::bail!("Failed to get labels/distances from RangeSearchResult");
            }

            let size = self.buffer_size();
            let labels = std::slice::from_raw_parts(labels_ptr, size);
            let distances = std::slice::from_raw_parts(distances_ptr, size);

            Ok((labels, distances))
        }
    }

    /// Get results for a specific query
    ///
    /// # Arguments
    /// * `query_idx` - Index of the query (0 to nq-1)
    ///
    /// # Returns
    /// A tuple of (labels, distances) for the given query
    pub fn get_query_results(&self, query_idx: usize) -> Result<(&[i64], &[f32])> {
        let nq = self.nq();
        if query_idx >= nq {
            anyhow::bail!("Query index {} out of range [0, {})", query_idx, nq);
        }

        let lims = self.lims()?;
        let (all_labels, all_distances) = self.labels_and_distances()?;

        let start = lims[query_idx];
        let end = lims[query_idx + 1];

        Ok((&all_labels[start..end], &all_distances[start..end]))
    }

    /// Get all results as a vector of (labels, distances) pairs, one per query
    ///
    /// # Returns
    /// A vector where each element is (Vec<i64>, Vec<f32>) for each query
    pub fn get_all_results(&self) -> Result<Vec<(Vec<i64>, Vec<f32>)>> {
        let lims = self.lims()?;
        let (all_labels, all_distances) = self.labels_and_distances()?;
        let nq = self.nq();

        let mut results = Vec::with_capacity(nq);
        for i in 0..nq {
            let start = lims[i];
            let end = lims[i + 1];

            let labels = all_labels[start..end].to_vec();
            let distances = all_distances[start..end].to_vec();

            results.push((labels, distances));
        }

        Ok(results)
    }

    /// Get the raw pointer (for use with FAISS functions)
    pub fn as_ptr(&self) -> *mut ffi::FaissRangeSearchResult {
        self.inner
    }
}

impl Drop for RangeSearchResult {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_RangeSearchResult_free(self.inner);
            }
        }
    }
}

unsafe impl Send for RangeSearchResult {}

/// BufferList - A dynamic buffer for storing variable-length results
///
/// BufferList is a growable buffer used to store results of variable size,
/// particularly useful for range search operations where the number of results
/// per query is not known in advance.
pub struct BufferList {
    inner: *mut ffi::FaissBufferList,
}

impl BufferList {
    /// Create a new BufferList with a given initial buffer size
    ///
    /// # Arguments
    /// * `buffer_size` - Initial size of each buffer
    pub fn new(buffer_size: usize) -> Result<Self> {
        unsafe {
            let mut bl = std::ptr::null_mut();
            let ret = ffi::faiss_BufferList_new(&mut bl, buffer_size);
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(Self { inner: bl })
        }
    }

    /// Get the current buffer size
    pub fn buffer_size(&self) -> usize {
        unsafe { ffi::faiss_BufferList_buffer_size(self.inner) }
    }

    /// Get the write pointer (number of elements written)
    pub fn wp(&self) -> usize {
        unsafe { ffi::faiss_BufferList_wp(self.inner) }
    }

    /// Append a new buffer to the list
    pub fn append_buffer(&mut self) -> Result<()> {
        unsafe {
            let ret = ffi::faiss_BufferList_append_buffer(self.inner);
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(())
        }
    }

    /// Add an element (distance, ID pair) to the buffer
    ///
    /// # Arguments
    /// * `dis` - Distance value
    /// * `id` - Vector ID
    pub fn add(&mut self, dis: f32, id: i64) -> Result<()> {
        unsafe {
            let ret = ffi::faiss_BufferList_add(self.inner, id, dis);
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(())
        }
    }

    /// Copy a range of elements to destination arrays
    ///
    /// # Arguments
    /// * `ofs` - Offset to start copying from
    /// * `n` - Number of elements to copy
    /// * `dest_ids` - Destination for IDs
    /// * `dest_dis` - Destination for distances
    pub fn copy_range(&self, ofs: usize, n: usize, dest_ids: &mut [i64], dest_dis: &mut [f32]) -> Result<()> {
        if dest_ids.len() < n || dest_dis.len() < n {
            anyhow::bail!("Destination arrays too small");
        }

        unsafe {
            let ret = ffi::faiss_BufferList_copy_range(
                self.inner,
                ofs,
                n,
                dest_ids.as_mut_ptr(),
                dest_dis.as_mut_ptr(),
            );
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(())
        }
    }

    /// Get the raw pointer (for use with FAISS functions)
    pub fn as_ptr(&self) -> *mut ffi::FaissBufferList {
        self.inner
    }
}

impl Drop for BufferList {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_BufferList_free(self.inner);
            }
        }
    }
}

unsafe impl Send for BufferList {}

/// RangeQueryResult - Result for a single query in a range search
///
/// Holds the (distance, ID) pairs found for one query within a given radius.
pub struct RangeQueryResult {
    inner: *mut ffi::FaissRangeQueryResult,
    // We don't own this pointer, it's managed by RangeSearchPartialResult
    _phantom: std::marker::PhantomData<ffi::FaissRangeQueryResult>,
}

impl RangeQueryResult {
    /// Create from a raw pointer (internal use)
    ///
    /// # Safety
    /// The pointer must be valid and the lifetime must be managed externally
    unsafe fn from_ptr(ptr: *mut ffi::FaissRangeQueryResult) -> Self {
        Self {
            inner: ptr,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the query number this result belongs to
    pub fn qno(&self) -> i64 {
        unsafe { ffi::faiss_RangeQueryResult_qno(self.inner) }
    }

    /// Get the number of results for this query
    pub fn nres(&self) -> usize {
        unsafe { ffi::faiss_RangeQueryResult_nres(self.inner) }
    }

    /// Add a result (distance, ID) to this query result
    ///
    /// # Arguments
    /// * `dis` - Distance value
    /// * `id` - Vector ID
    pub fn add(&mut self, dis: f32, id: i64) -> Result<()> {
        unsafe {
            let ret = ffi::faiss_RangeQueryResult_add(self.inner, dis, id);
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(())
        }
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *mut ffi::FaissRangeQueryResult {
        self.inner
    }
}

/// RangeSearchPartialResult - Partial results for multi-threaded range search
///
/// Used to accumulate results from multiple threads before finalizing them
/// into a RangeSearchResult. Each thread can work on its own partial result
/// and they are merged at the end.
pub struct RangeSearchPartialResult {
    inner: *mut ffi::FaissRangeSearchPartialResult,
}

impl RangeSearchPartialResult {
    /// Create a new RangeSearchPartialResult
    ///
    /// # Arguments
    /// * `result` - The RangeSearchResult to eventually merge into
    pub fn new(result: &mut RangeSearchResult) -> Result<Self> {
        unsafe {
            let mut partial = std::ptr::null_mut();
            let ret = ffi::faiss_RangeSearchPartialResult_new(&mut partial, result.as_ptr());
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(Self { inner: partial })
        }
    }

    /// Create a new result for a specific query
    ///
    /// # Arguments
    /// * `qno` - Query number
    ///
    /// # Returns
    /// A RangeQueryResult for accumulating this query's results
    pub fn new_result(&mut self, qno: i64) -> Result<RangeQueryResult> {
        unsafe {
            let mut qr = std::ptr::null_mut();
            let ret = ffi::faiss_RangeSearchPartialResult_new_result(self.inner, qno, &mut qr);
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(RangeQueryResult::from_ptr(qr))
        }
    }

    /// Set the lims array (CSR boundaries) for the results
    pub fn set_lims(&mut self) -> Result<()> {
        unsafe {
            let ret = ffi::faiss_RangeSearchPartialResult_set_lims(self.inner);
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(())
        }
    }

    /// Finalize the partial results and merge into the main result
    pub fn finalize(&mut self) -> Result<()> {
        unsafe {
            let ret = ffi::faiss_RangeSearchPartialResult_finalize(self.inner);
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(())
        }
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *mut ffi::FaissRangeSearchPartialResult {
        self.inner
    }
}

// Note: RangeSearchPartialResult is typically freed by FAISS internally
// when finalized, so we don't implement Drop

unsafe impl Send for RangeSearchPartialResult {}
