use std::ptr;

use crate::bindings;
use crate::error::{check_return_code, Result};

pub struct IDSelector {
    pub(crate) inner: *mut bindings::FaissIDSelector,
}

impl Drop for IDSelector {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { bindings::faiss_IDSelector_free(self.inner) }
        }
    }
}

pub struct IDSelectorRange {
    pub(crate) inner: *mut bindings::FaissIDSelectorRange,
}

impl Drop for IDSelectorRange {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { bindings::faiss_IDSelectorRange_free(self.inner) }
        }
    }
}

impl IDSelectorRange {
    pub fn new(imin: i64, imax: i64) -> Result<Self> {
        let mut inner = ptr::null_mut();
        check_return_code(unsafe { bindings::faiss_IDSelectorRange_new(&mut inner, imin, imax) })?;
        Ok(Self { inner })
    }
}

pub struct RangeSearchResult {
    pub(crate) inner: *mut bindings::FaissRangeSearchResult,
}

impl Drop for RangeSearchResult {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { bindings::faiss_RangeSearchResult_free(self.inner) }
        }
    }
}

impl RangeSearchResult {
    pub fn new(nq: i64) -> Result<Self> {
        let mut inner = ptr::null_mut();
        check_return_code(unsafe { bindings::faiss_RangeSearchResult_new(&mut inner, nq) })?;
        Ok(Self { inner })
    }

    pub fn nq(&self) -> usize {
        unsafe { bindings::faiss_RangeSearchResult_nq(self.inner) }
    }

    pub fn buffer_size(&self) -> usize {
        unsafe { bindings::faiss_RangeSearchResult_buffer_size(self.inner) }
    }

    pub fn lims(&mut self) -> Vec<usize> {
        let mut ptr = ptr::null_mut();
        unsafe { bindings::faiss_RangeSearchResult_lims(self.inner, &mut ptr) };
        let nq = self.nq();
        unsafe { std::slice::from_raw_parts(ptr, nq + 1).to_vec() }
    }

    pub fn labels_and_distances(&mut self) -> (Vec<i64>, Vec<f32>) {
        let mut labels = ptr::null_mut();
        let mut distances = ptr::null_mut();
        unsafe {
            bindings::faiss_RangeSearchResult_labels(self.inner, &mut labels, &mut distances)
        };
        let lims = self.lims();
        let n = lims.last().copied().unwrap_or(0);
        let labels_vec = unsafe { std::slice::from_raw_parts(labels, n) }.to_vec();
        let distances_vec = unsafe { std::slice::from_raw_parts(distances, n) }.to_vec();
        (labels_vec, distances_vec)
    }
}
