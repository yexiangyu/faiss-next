use crate::error::FaissError;
use crate::ffi;
use crate::index::{IndexOwned, IndexTrait};
use anyhow::Result;

/// Index that applies transformations to vectors before indexing
///
/// IndexPreTransform wraps another index and applies one or more vector
/// transformations (like PCA, OPQ, rotation) before passing vectors to
/// the underlying index. This is useful for dimensionality reduction or
/// improving index quality.
///
/// # How it Works
/// 1. **Transform**: Apply transformations to input vectors
/// 2. **Index**: Pass transformed vectors to the sub-index
/// 3. **Search**: Transform query vectors then search in sub-index
///
/// # Common Transformations
/// - **PCA**: Dimensionality reduction
/// - **OPQ**: Optimized Product Quantization
/// - **Random Rotation**: Improve data distribution
///
/// # When to Use
/// - Need to reduce vector dimensionality before indexing
/// - Want to apply learned transformations (PCA, OPQ)
/// - Combine multiple preprocessing steps
///
/// # Example
/// ```ignore
/// use faiss_next::{IndexPreTransform, IndexFlatL2};
///
/// // Create base index
/// let base_index = IndexFlatL2::new(64)?;
///
/// // Wrap with pre-transform (will apply PCA 128->64)
/// let mut index = IndexPreTransform::new_with(&base_index)?;
///
/// // Add PCA transform (requires VectorTransform)
/// // index.prepend_transform(&pca)?;
///
/// // Train and use normally
/// index.train(1000, &training_data)?;
/// index.add(1000, &vectors)?;
/// ```
pub struct IndexPreTransform {
    inner: *mut ffi::FaissIndexPreTransform,
}

impl IndexPreTransform {
    /// Create a new IndexPreTransform (requires manual setup)
    ///
    /// # Note
    /// You probably want to use `new_with` instead.
    pub fn new() -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexPreTransform_new(&mut index_ptr);

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Create a new IndexPreTransform with a sub-index
    ///
    /// # Arguments
    /// * `index` - The underlying index to wrap
    ///
    /// # Example
    /// ```ignore
    /// use faiss_next::{IndexPreTransform, IndexFlatL2};
    ///
    /// let base_index = IndexFlatL2::new(128)?;
    /// let index = IndexPreTransform::new_with(&base_index)?;
    /// ```
    pub fn new_with(index: &impl IndexTrait) -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexPreTransform_new_with(
                &mut index_ptr,
                index.inner_ptr(),
            );

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Create a new IndexPreTransform with transform and index
    ///
    /// # Arguments
    /// * `transform` - The vector transformation to apply
    /// * `index` - The underlying index
    ///
    /// # Note
    /// Requires a VectorTransform object. Use `new_with` and `prepend_transform`
    /// for more flexibility.
    ///
    /// # Safety
    /// The caller must ensure that `transform` is a valid pointer to a FaissVectorTransform.
    pub unsafe fn new_with_transform(
        transform: *mut ffi::FaissVectorTransform,
        index: &impl IndexTrait,
    ) -> Result<Self> {
        unsafe {
            let mut index_ptr = std::ptr::null_mut();
            let ret = ffi::faiss_IndexPreTransform_new_with_transform(
                &mut index_ptr,
                transform,
                index.inner_ptr(),
            );

            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }

            Ok(Self { inner: index_ptr })
        }
    }

    /// Cast a generic index to IndexPreTransform
    ///
    /// # Arguments
    /// * `index` - The index to cast
    ///
    /// # Returns
    /// `Some(IndexPreTransform)` if the index is actually a PreTransform index, `None` otherwise
    pub fn from_index(index: &IndexOwned) -> Option<Self> {
        unsafe {
            let ptr = ffi::faiss_IndexPreTransform_cast(index.as_ptr());
            if ptr.is_null() {
                None
            } else {
                Some(Self { inner: ptr })
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

    /// Add a transformation to the beginning of the chain
    ///
    /// Transformations are applied in the order they are prepended.
    /// The last prepended transform is applied first.
    ///
    /// # Arguments
    /// * `transform` - The vector transformation to add
    ///
    /// # Example
    /// ```ignore
    /// // Create transforms (requires VectorTransform module)
    /// let pca = VectorTransform::pca(128, 64)?;
    /// let rotation = VectorTransform::random_rotation(64)?;
    ///
    /// // Add transforms (applied in reverse order: rotation then PCA)
    /// index.prepend_transform(&pca)?;
    /// index.prepend_transform(&rotation)?;
    /// ```
    ///
    /// # Safety
    /// The caller must ensure that `transform` is a valid pointer to a FaissVectorTransform.
    pub unsafe fn prepend_transform(&mut self, transform: *mut ffi::FaissVectorTransform) -> Result<()> {
        unsafe {
            let ret = ffi::faiss_IndexPreTransform_prepend_transform(self.inner, transform);
            if let Some(err) = FaissError::from_code(ret) {
                return Err(err.into());
            }
        }
        Ok(())
    }

    /// Get the underlying index
    ///
    /// Returns a raw pointer to the sub-index.
    pub fn index(&self) -> *mut ffi::FaissIndex {
        unsafe { ffi::faiss_IndexPreTransform_index(self.inner) }
    }

    /// Check if the index owns its internal fields
    ///
    /// If true, the index will free the sub-index and transforms when dropped.
    pub fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexPreTransform_own_fields(self.inner) != 0 }
    }

    /// Set whether the index owns its internal fields
    ///
    /// # Arguments
    /// * `own` - If true, index will manage sub-index and transform lifetimes
    ///
    /// # Safety
    /// Setting this to false means you must manually manage the lifetime
    /// of the sub-index and transforms.
    pub fn set_own_fields(&mut self, own: bool) {
        unsafe {
            ffi::faiss_IndexPreTransform_set_own_fields(self.inner, own as i32);
        }
    }

    /// Get the inner pointer
    pub fn as_ptr(&self) -> *mut ffi::FaissIndexPreTransform {
        self.inner
    }
}

impl Drop for IndexPreTransform {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                ffi::faiss_IndexPreTransform_free(self.inner);
            }
        }
    }
}

impl IndexTrait for IndexPreTransform {
    fn inner_ptr(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut ffi::FaissIndex
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index_flat::IndexFlatL2;

    #[test]
    fn test_index_pre_transform_creation() {
        let index = IndexPreTransform::new();
        assert!(index.is_ok(), "Failed to create IndexPreTransform");
    }

    #[test]
    fn test_index_pre_transform_with_index() {
        let base_index = IndexFlatL2::new(128).unwrap();
        let base_owned = base_index.into_index().unwrap();
        let index = IndexPreTransform::new_with(&base_owned);
        assert!(index.is_ok(), "Failed to create IndexPreTransform with index");

        let index = index.unwrap();
        assert_eq!(index.d().unwrap(), 128);

        // Check that it has a sub-index
        assert!(!index.index().is_null());
    }

    #[test]
    fn test_index_pre_transform_own_fields() {
        let base_index = IndexFlatL2::new(64).unwrap();
        let base_owned = base_index.into_index().unwrap();
        let index = IndexPreTransform::new_with(&base_owned).unwrap();

        // Check own_fields works (returns a boolean value)
        let _owns = index.own_fields();
    }

    #[test]
    fn test_index_pre_transform_basic_operations() {
        // Just test creation without crashing
        let index = IndexPreTransform::new().unwrap();
        assert!(!index.inner_ptr().is_null());
    }
}
