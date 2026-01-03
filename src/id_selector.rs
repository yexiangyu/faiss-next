use crate::error::FaissError;
use crate::ffi;
use anyhow::Result;

/// A trait for ID selectors used to select which vectors to remove from an index
pub trait IDSelectorTrait {
    /// Get the underlying raw pointer to the FAISS IDSelector
    fn as_ptr(&self) -> *const ffi::FaissIDSelector;
}

/// An ID selector that selects a batch of specific IDs
pub struct IDSelectorBatch {
    inner: *mut ffi::FaissIDSelectorBatch,
}

impl IDSelectorBatch {
    /// Create a new IDSelectorBatch from a slice of IDs
    ///
    /// # Arguments
    /// * `ids` - The IDs to select
    pub fn new(ids: &[i64]) -> Result<Self> {
        unsafe {
            let mut selector = std::ptr::null_mut();
            let ret = ffi::faiss_IDSelectorBatch_new(
                &mut selector,
                ids.len(),
                ids.as_ptr(),
            );
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
            Ok(Self { inner: selector })
        }
    }
}

impl Drop for IDSelectorBatch {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IDSelector_free(self.inner as *mut ffi::FaissIDSelector);
            }
        }
    }
}

impl IDSelectorTrait for IDSelectorBatch {
    fn as_ptr(&self) -> *const ffi::FaissIDSelector {
        self.inner as *const ffi::FaissIDSelector
    }
}

/// An ID selector that selects a range of IDs
pub struct IDSelectorRange {
    inner: *mut ffi::FaissIDSelectorRange,
}

impl IDSelectorRange {
    /// Create a new IDSelectorRange for IDs in [imin, imax)
    ///
    /// # Arguments
    /// * `imin` - Minimum ID (inclusive)
    /// * `imax` - Maximum ID (exclusive)
    pub fn new(imin: i64, imax: i64) -> Result<Self> {
        unsafe {
            let mut selector = std::ptr::null_mut();
            let ret = ffi::faiss_IDSelectorRange_new(&mut selector, imin, imax);
            if ret != 0 {
                anyhow::bail!("Failed to create IDSelectorRange");
            }
            Ok(Self { inner: selector })
        }
    }
}

impl Drop for IDSelectorRange {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IDSelector_free(self.inner as *mut ffi::FaissIDSelector);
            }
        }
    }
}

impl IDSelectorTrait for IDSelectorRange {
    fn as_ptr(&self) -> *const ffi::FaissIDSelector {
        self.inner as *const ffi::FaissIDSelector
    }
}

/// An ID selector that uses a bitmap to select IDs
pub struct IDSelectorBitmap {
    inner: *mut ffi::FaissIDSelectorBitmap,
}

impl IDSelectorBitmap {
    /// Create a new IDSelectorBitmap
    ///
    /// # Arguments
    /// * `n` - Number of elements in the bitmap
    /// * `bitmap` - Bitmap array where each byte represents 8 IDs
    pub fn new(n: usize, bitmap: &[u8]) -> Result<Self> {
        unsafe {
            let mut selector = std::ptr::null_mut();
            let ret = ffi::faiss_IDSelectorBitmap_new(&mut selector, n, bitmap.as_ptr());
            if ret != 0 {
                anyhow::bail!("Failed to create IDSelectorBitmap");
            }
            Ok(Self { inner: selector })
        }
    }
}

impl Drop for IDSelectorBitmap {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IDSelector_free(self.inner as *mut ffi::FaissIDSelector);
            }
        }
    }
}

impl IDSelectorTrait for IDSelectorBitmap {
    fn as_ptr(&self) -> *const ffi::FaissIDSelector {
        self.inner as *const ffi::FaissIDSelector
    }
}

/// An ID selector that inverts another selector (NOT operation)
pub struct IDSelectorNot {
    inner: *mut ffi::FaissIDSelectorNot,
    // Keep the child selector alive
    _child: Box<dyn IDSelectorTrait>,
}

impl IDSelectorNot {
    /// Create a new IDSelectorNot that inverts the given selector
    ///
    /// # Arguments
    /// * `selector` - The selector to invert
    pub fn new(selector: Box<dyn IDSelectorTrait>) -> Result<Self> {
        unsafe {
            let mut not_selector = std::ptr::null_mut();
            let ret = ffi::faiss_IDSelectorNot_new(&mut not_selector, selector.as_ptr());
            if ret != 0 {
                anyhow::bail!("Failed to create IDSelectorNot");
            }
            Ok(Self {
                inner: not_selector,
                _child: selector,
            })
        }
    }
}

impl Drop for IDSelectorNot {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IDSelector_free(self.inner as *mut ffi::FaissIDSelector);
            }
        }
    }
}

impl IDSelectorTrait for IDSelectorNot {
    fn as_ptr(&self) -> *const ffi::FaissIDSelector {
        self.inner as *const ffi::FaissIDSelector
    }
}

/// An ID selector that combines two selectors with AND operation
pub struct IDSelectorAnd {
    inner: *mut ffi::FaissIDSelectorAnd,
    // Keep the child selectors alive
    _lhs: Box<dyn IDSelectorTrait>,
    _rhs: Box<dyn IDSelectorTrait>,
}

impl IDSelectorAnd {
    /// Create a new IDSelectorAnd that combines two selectors with AND
    ///
    /// # Arguments
    /// * `lhs` - Left-hand side selector
    /// * `rhs` - Right-hand side selector
    pub fn new(lhs: Box<dyn IDSelectorTrait>, rhs: Box<dyn IDSelectorTrait>) -> Result<Self> {
        unsafe {
            let mut and_selector = std::ptr::null_mut();
            let ret = ffi::faiss_IDSelectorAnd_new(&mut and_selector, lhs.as_ptr(), rhs.as_ptr());
            if ret != 0 {
                anyhow::bail!("Failed to create IDSelectorAnd");
            }
            Ok(Self {
                inner: and_selector,
                _lhs: lhs,
                _rhs: rhs,
            })
        }
    }
}

impl Drop for IDSelectorAnd {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IDSelector_free(self.inner as *mut ffi::FaissIDSelector);
            }
        }
    }
}

impl IDSelectorTrait for IDSelectorAnd {
    fn as_ptr(&self) -> *const ffi::FaissIDSelector {
        self.inner as *const ffi::FaissIDSelector
    }
}

/// An ID selector that combines two selectors with OR operation
pub struct IDSelectorOr {
    inner: *mut ffi::FaissIDSelectorOr,
    // Keep the child selectors alive
    _lhs: Box<dyn IDSelectorTrait>,
    _rhs: Box<dyn IDSelectorTrait>,
}

impl IDSelectorOr {
    /// Create a new IDSelectorOr that combines two selectors with OR
    ///
    /// # Arguments
    /// * `lhs` - Left-hand side selector
    /// * `rhs` - Right-hand side selector
    pub fn new(lhs: Box<dyn IDSelectorTrait>, rhs: Box<dyn IDSelectorTrait>) -> Result<Self> {
        unsafe {
            let mut or_selector = std::ptr::null_mut();
            let ret = ffi::faiss_IDSelectorOr_new(&mut or_selector, lhs.as_ptr(), rhs.as_ptr());
            if ret != 0 {
                anyhow::bail!("Failed to create IDSelectorOr");
            }
            Ok(Self {
                inner: or_selector,
                _lhs: lhs,
                _rhs: rhs,
            })
        }
    }
}

impl Drop for IDSelectorOr {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IDSelector_free(self.inner as *mut ffi::FaissIDSelector);
            }
        }
    }
}

impl IDSelectorTrait for IDSelectorOr {
    fn as_ptr(&self) -> *const ffi::FaissIDSelector {
        self.inner as *const ffi::FaissIDSelector
    }
}

/// An ID selector that combines two selectors with XOR operation
pub struct IDSelectorXor {
    inner: *mut ffi::FaissIDSelectorXOr,
    // Keep the child selectors alive
    _lhs: Box<dyn IDSelectorTrait>,
    _rhs: Box<dyn IDSelectorTrait>,
}

impl IDSelectorXor {
    /// Create a new IDSelectorXor that combines two selectors with XOR
    ///
    /// # Arguments
    /// * `lhs` - Left-hand side selector
    /// * `rhs` - Right-hand side selector
    pub fn new(lhs: Box<dyn IDSelectorTrait>, rhs: Box<dyn IDSelectorTrait>) -> Result<Self> {
        unsafe {
            let mut xor_selector = std::ptr::null_mut();
            let ret = ffi::faiss_IDSelectorXOr_new(&mut xor_selector, lhs.as_ptr(), rhs.as_ptr());
            if ret != 0 {
                anyhow::bail!("Failed to create IDSelectorXor");
            }
            Ok(Self {
                inner: xor_selector,
                _lhs: lhs,
                _rhs: rhs,
            })
        }
    }
}

impl Drop for IDSelectorXor {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IDSelector_free(self.inner as *mut ffi::FaissIDSelector);
            }
        }
    }
}

impl IDSelectorTrait for IDSelectorXor {
    fn as_ptr(&self) -> *const ffi::FaissIDSelector {
        self.inner as *const ffi::FaissIDSelector
    }
}
