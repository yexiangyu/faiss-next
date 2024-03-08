use itertools::Itertools;
use std::ptr::{addr_of_mut, null_mut};

use crate::error::Result;
use crate::rc;
use faiss_next_sys as sys;
use tracing::trace;

pub struct FaissRangeSearchResult {
    pub inner: *mut sys::FaissRangeSearchResult,
}

impl Drop for FaissRangeSearchResult {
    fn drop(&mut self) {
        unsafe { sys::faiss_RangeSearchResult_free(self.inner) }
        trace!("drop range search result={:?}", self.inner);
    }
}

impl FaissRangeSearchResult {
    pub fn new(nq: i64) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_RangeSearchResult_new(addr_of_mut!(inner), nq) })?;
        Ok(Self { inner })
    }

    pub fn new_with(nq: i64, alloc_lims: bool) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_RangeSearchResult_new_with(addr_of_mut!(inner), nq, alloc_lims as i32) })?;
        Ok(Self { inner })
    }

    pub fn do_allocation(&mut self) -> Result<()> {
        rc!({ sys::faiss_RangeSearchResult_do_allocation(self.inner) })?;
        Ok(())
    }

    pub fn buffer_size(&self) -> usize {
        unsafe { sys::faiss_RangeSearchResult_buffer_size(self.inner) }
    }

    pub fn lims(&self) -> &[usize] {
        let mut lims = null_mut();
        let nq = self.nq();
        unsafe { sys::faiss_RangeSearchResult_lims(self.inner, addr_of_mut!(lims)) }
        unsafe { std::slice::from_raw_parts(lims, nq + 1) }
    }

    pub fn nq(&self) -> usize {
        unsafe { sys::faiss_RangeSearchResult_nq(self.inner) }
    }

    pub fn labels(&self) -> Vec<(&[i64], &[f32])> {
        let mut labels = null_mut();
        let mut distances = null_mut();
        unsafe {
            sys::faiss_RangeSearchResult_labels(
                self.inner,
                addr_of_mut!(labels),
                addr_of_mut!(distances),
            )
        }
        self.lims()
            .iter()
            .tuple_windows()
            .map(|(a, b)| {
                let labels = unsafe { std::slice::from_raw_parts(labels.add(*a), *b - *a) };
                let distances = unsafe { std::slice::from_raw_parts(distances.add(*a), *b - *a) };
                (labels, distances)
            })
            .collect()
    }
}
