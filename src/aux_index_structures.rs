#[cxx::bridge]
#[allow(clippy::missing_safety_doc)]
pub mod ffi {
    unsafe extern "C++" {
        include!("faiss-next/src/cpp/aux_index_structures.hpp");
        unsafe fn range_search_result_new(nq: usize, alloc_lims: bool) -> *mut i32;
        unsafe fn range_search_result_free(ptr: *mut i32);
        unsafe fn range_search_result_do_allocation(ptr: *mut i32);
        unsafe fn range_search_result_nq(ptr: *const i32) -> usize;
        unsafe fn range_search_result_lims(ptr: *const i32) -> *const usize;
        unsafe fn range_search_result_labels(ptr: *const i32) -> *const i64;
        unsafe fn range_search_result_distances(ptr: *const i32) -> *const f32;
        unsafe fn range_search_result_buffer_size(ptr: *const i32) -> usize;
    }
}

use std::slice::from_raw_parts;

use tracing::*;

pub type RangeSearchResultPtr = *mut i32;

pub struct RangeSearchResult {
    inner: RangeSearchResultPtr,
}

impl Drop for RangeSearchResult {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            trace!("drop range_search_result inner={:?}", self.inner);
            unsafe { ffi::range_search_result_free(self.inner) }
        }
    }
}

impl RangeSearchResult {
    pub fn ptr(&mut self) -> RangeSearchResultPtr {
        self.inner
    }

    pub fn new(nq: usize, alloc_lims: bool) -> Self {
        let inner = unsafe { ffi::range_search_result_new(nq, alloc_lims) };
        trace!(%nq, %alloc_lims, "create range_search_result inner={:?}", inner);
        RangeSearchResult { inner }
    }
    pub fn do_allocation(&mut self) {
        unsafe { ffi::range_search_result_do_allocation(self.inner) }
    }

    pub fn nq(&self) -> usize {
        unsafe { ffi::range_search_result_nq(self.inner) }
    }

    pub fn lims(&self) -> &[usize] {
        let lims = unsafe { ffi::range_search_result_lims(self.inner) };
        let nq = self.nq();
        unsafe { from_raw_parts(lims, nq + 1) }
    }

    pub fn labels(&self) -> Vec<&[i64]> {
        let labels = unsafe { ffi::range_search_result_labels(self.inner) };
        let lims = self.lims();
        lims.windows(2)
            .map(|idx| {
                let start = idx[0];
                let end = idx[1];
                let labels = unsafe { from_raw_parts(labels.add(start), end - start) };
                labels
            })
            .collect()
    }

    pub fn distances(&self) -> Vec<&[f32]> {
        let distances = unsafe { ffi::range_search_result_distances(self.inner) };
        let lims = self.lims();
        lims.windows(2)
            .map(|idx| {
                let start = idx[0];
                let end = idx[1];
                let distances = unsafe { from_raw_parts(distances.add(start), end - start) };
                distances
            })
            .collect()
    }

    pub fn buffer_size(&self) -> usize {
        unsafe { ffi::range_search_result_buffer_size(self.inner) }
    }
}
