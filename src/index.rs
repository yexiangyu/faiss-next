use std::ptr;

use crate::bindings;
use crate::error::{check_return_code, Result};
use crate::macros::*;
use crate::traits::FaissIndex;

pub struct Index {
    pub(crate) inner: *mut bindings::FaissIndex,
}

impl_faiss_drop!(Index, faiss_Index_free);
impl_index_common!(Index);

impl FaissIndex for Index {
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

pub fn clone_index(source: &Index) -> Result<Index> {
    let mut inner = ptr::null_mut();
    check_return_code(unsafe { bindings::faiss_clone_index(source.inner, &mut inner) })?;
    Ok(Index { inner })
}
