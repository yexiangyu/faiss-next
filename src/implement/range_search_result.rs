use std::ptr::null_mut;

use faiss_next_sys as sys;
use tracing::trace;

use crate::{error::Result, macros::rc};

pub struct RangeSearchResult {
    inner: *mut sys::FaissRangeSearchResult,
}

impl Drop for RangeSearchResult {
    fn drop(&mut self) {
        trace!(?self, "drop");
        unsafe { sys::faiss_RangeSearchResult_free(self.inner) }
    }
}

impl RangeSearchResult {
    pub fn ptr(&self) -> *mut sys::FaissRangeSearchResult {
        self.inner
    }

    pub fn new(nq: i64, alloc_lims: bool) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_RangeSearchResult_new_with(&mut inner, nq, alloc_lims as i32) })?;
        let r = Self { inner };
        trace!(?r, "new");
        Ok(r)
    }

    pub fn do_allocation(&mut self) -> Result<()> {
        rc!({ sys::faiss_RangeSearchResult_do_allocation(self.inner) })
    }

    pub fn buffer_size(&self) -> usize {
        unsafe { sys::faiss_RangeSearchResult_buffer_size(self.inner) }
    }

    pub fn lims(&self) -> &[usize] {
        let mut lims = null_mut();
        unsafe { sys::faiss_RangeSearchResult_lims(self.ptr(), &mut lims) };
        unsafe { std::slice::from_raw_parts(lims, self.nq() + 1) }
    }

    pub fn nq(&self) -> usize {
        unsafe { sys::faiss_RangeSearchResult_nq(self.ptr()) }
    }

    pub fn labels(&self) -> Vec<(&[i64], &[f32])> {
        let mut labels = null_mut();
        let mut distances = null_mut();
        unsafe { sys::faiss_RangeSearchResult_labels(self.ptr(), &mut labels, &mut distances) };
        self.lims()
            .windows(2)
            .map(|se| unsafe {
                let offset = se[0];
                let len = se[1] - se[0];
                let l = labels.add(offset);
                let d = distances.add(offset);
                let l = std::slice::from_raw_parts(l, len);
                let d = std::slice::from_raw_parts(d, len);
                (l, d)
            })
            .collect()
    }
}

impl std::fmt::Debug for RangeSearchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RangeSearchResult")
            .field("inner", &self.inner)
            .field("nq", &self.nq())
            .field("buffer_size", &self.buffer_size())
            .field("lims", &self.lims())
            .finish()
    }
}

// Buffer?

pub struct BufferList {
    inner: *mut sys::FaissBufferList,
}

impl Drop for BufferList {
    fn drop(&mut self) {
        unsafe { sys::faiss_BufferList_free(self.inner) }
    }
}

impl BufferList {
    pub fn new(buffer_size: usize) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_BufferList_new(&mut inner, buffer_size) })?;
        Ok(Self { inner })
    }

    pub fn buffer_size(&self) -> usize {
        unsafe { sys::faiss_BufferList_buffer_size(self.inner) }
    }

    pub fn wp(&self) -> usize {
        unsafe { sys::faiss_BufferList_wp(self.inner) }
    }

    pub fn append_buffer(&mut self) -> Result<()> {
        rc!({ sys::faiss_BufferList_append_buffer(self.inner) })
    }

    pub fn add(&mut self, id: i64, dis: f32) -> Result<()> {
        rc!({ sys::faiss_BufferList_add(self.inner, id, dis) })
    }

    pub fn copy_range(
        &self,
        offset: usize,
        mut dest_ids: impl AsMut<[i64]>,
        mut dest_dis: impl AsMut<[f32]>,
    ) -> Result<()> {
        let dest_ids = dest_ids.as_mut();
        let dest_dis = dest_dis.as_mut();
        let n = dest_ids.len();
        rc!({
            sys::faiss_BufferList_copy_range(
                self.inner,
                offset,
                n,
                dest_ids.as_mut_ptr(),
                dest_dis.as_mut_ptr(),
            )
        })
    }
}

pub struct RangeSearchPartialResult {
    inner: *mut sys::FaissRangeSearchPartialResult,
}

impl RangeSearchPartialResult {
    pub fn new(res_in: RangeSearchResult) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_RangeSearchPartialResult_new(&mut inner, res_in.ptr()) })?;
        Ok(Self { inner })
    }

    pub fn res(&self) -> RangeSearchResult {
        let inner = unsafe { sys::faiss_RangeSearchPartialResult_res(self.inner) };
        RangeSearchResult { inner }
    }

    pub fn finalize(&mut self) -> Result<()> {
        rc!({ sys::faiss_RangeSearchPartialResult_finalize(self.inner) })
    }

    pub fn set_lims(&mut self) -> Result<()> {
        rc!({ sys::faiss_RangeSearchPartialResult_set_lims(self.inner) })
    }

    pub fn new_result(&self, qno: i64) -> Result<RangeQueryResult> {
        let mut inner = null_mut();
        rc!({ sys::faiss_RangeSearchPartialResult_new_result(self.inner, qno, &mut inner) })?;
        Ok(RangeQueryResult { inner })
    }
}

pub struct RangeQueryResult {
    inner: *mut sys::FaissRangeQueryResult,
}

impl RangeQueryResult {
    pub fn qno(&self) -> i64 {
        unsafe { sys::faiss_RangeQueryResult_qno(self.inner) }
    }

    pub fn nres(&self) -> usize {
        unsafe { sys::faiss_RangeQueryResult_nres(self.inner) }
    }

    pub fn pres(&self) -> RangeSearchPartialResult {
        let inner = unsafe { sys::faiss_RangeQueryResult_pres(self.inner) };
        RangeSearchPartialResult { inner }
    }

    pub fn add(&mut self, dis: f32, id: i64) -> Result<()> {
        rc!({ sys::faiss_RangeQueryResult_add(self.inner, dis, id) })
    }
}
