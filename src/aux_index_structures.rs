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

pub trait IDSelectorTrait {
    fn ptr(&self) -> *mut sys::FaissIDSelector;

    fn is_member(&self, id: i64) -> bool {
        unsafe { sys::faiss_IDSelector_is_member(self.ptr(), id) != 0 }
    }

    fn not(self) -> Result<IDSelectorNot>
    where
        Self: Sized + 'static,
    {
        let source = Box::from(self);
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorNot_new(&mut inner, source.ptr()) })?;
        trace!(
            "create IDSelectorNot inner={:?}, source={:?}",
            inner,
            source.ptr()
        );
        Ok(IDSelectorNot { inner, source })
    }

    fn and(self, rhs: impl IDSelectorTrait + 'static) -> Result<IDSelectorAnd>
    where
        Self: Sized + 'static,
    {
        let l = Box::from(self);
        let r = Box::from(rhs);
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorAnd_new(&mut inner, l.ptr(), r.ptr()) })?;
        trace!(
            "create IDSelectorAnd inner={:?}, l={:?}, r={:?}",
            inner,
            l.ptr(),
            r.ptr()
        );
        Ok(IDSelectorAnd { inner, l, r })
    }

    fn or(self, rhs: impl IDSelectorTrait + 'static) -> Result<IDSelectorOr>
    where
        Self: Sized + 'static,
    {
        let l = Box::from(self);
        let r = Box::from(rhs);
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorOr_new(&mut inner, l.ptr(), r.ptr()) })?;
        trace!(
            "create IDSelectorOr inner={:?}, l={:?}, r={:?}",
            inner,
            l.ptr(),
            r.ptr()
        );
        Ok(IDSelectorOr { inner, l, r })
    }

    fn xor(self, rhs: impl IDSelectorTrait + 'static) -> Result<IDSelectorXOr>
    where
        Self: Sized + 'static,
    {
        let l = Box::from(self);
        let r = Box::from(rhs);
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorXOr_new(&mut inner, l.ptr(), r.ptr()) })?;
        trace!(
            "create IDSelectorXOr inner={:?}, l={:?}, r={:?}",
            inner,
            l.ptr(),
            r.ptr()
        );
        Ok(IDSelectorXOr { inner, l, r })
    }
}

macro_rules! impl_id_selector {
    ($cls: ty) => {
        impl IDSelectorTrait for $cls {
            fn ptr(&self) -> *mut sys::FaissIDSelector {
                self.inner as *mut _
            }
        }
    };
}

macro_rules! impl_drop {
    ($cls: ty, $free: ident) => {
        impl Drop for $cls {
            fn drop(&mut self) {
                tracing::trace!("drop {} inner={:?}", stringify!($cls), self.inner);
                unsafe { sys::$free(self.inner as *mut _) }
            }
        }
    };
}

pub struct IDSelectorRange {
    inner: *mut sys::FaissIDSelectorRange,
}
impl_drop!(IDSelectorRange, faiss_IDSelectorRange_free);
impl_id_selector!(IDSelectorRange);

impl std::fmt::Debug for IDSelectorRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IDSelectorRange")
            .field("inner", &self.inner)
            .finish()
    }
}

impl IDSelectorRange {
    pub fn new(imin: i64, imax: i64) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorRange_new(&mut inner, imin, imax) })?;
        let r = Self { inner };
        trace!(?r, %imin, %imax, "create");
        Ok(r)
    }
}

pub struct IDSelectorBatch {
    inner: *mut sys::FaissIDSelectorBatch,
}
impl_drop!(IDSelectorBatch, faiss_IDSelector_free);
impl_id_selector!(IDSelectorBatch);

impl std::fmt::Debug for IDSelectorBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IDSelectorBatch")
            .field("inner", &self.inner)
            .finish()
    }
}

impl IDSelectorBatch {
    pub fn new(ids: impl AsRef<[i64]>) -> Result<Self> {
        let mut inner = null_mut();
        let ids = ids.as_ref();
        let n = ids.len();
        trace!("create IDSelectorBatch n={}", n);
        rc!({ sys::faiss_IDSelectorBatch_new(&mut inner, n, ids.as_ptr()) })?;
        let r = Self { inner };
        trace!(?r, "create");
        Ok(r)
    }
}

pub struct IDSelectorNot {
    inner: *mut sys::FaissIDSelectorNot,
    #[allow(unused)]
    source: Box<dyn IDSelectorTrait>,
}
impl_drop!(IDSelectorNot, faiss_IDSelector_free);
impl_id_selector!(IDSelectorNot);

pub struct IDSelectorAnd {
    inner: *mut sys::FaissIDSelectorAnd,
    #[allow(unused)]
    l: Box<dyn IDSelectorTrait>,
    #[allow(unused)]
    r: Box<dyn IDSelectorTrait>,
}
impl_drop!(IDSelectorAnd, faiss_IDSelector_free);
impl_id_selector!(IDSelectorAnd);

pub struct IDSelectorOr {
    inner: *mut sys::FaissIDSelectorOr,
    #[allow(unused)]
    l: Box<dyn IDSelectorTrait>,
    #[allow(unused)]
    r: Box<dyn IDSelectorTrait>,
}
impl_drop!(IDSelectorOr, faiss_IDSelector_free);
impl_id_selector!(IDSelectorOr);

pub struct IDSelectorXOr {
    inner: *mut sys::FaissIDSelectorXOr,
    #[allow(unused)]
    l: Box<dyn IDSelectorTrait>,
    #[allow(unused)]
    r: Box<dyn IDSelectorTrait>,
}
impl_drop!(IDSelectorXOr, faiss_IDSelector_free);
impl_id_selector!(IDSelectorXOr);

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

pub trait DistanceComputerTrait {
    fn ptr(&self) -> *mut sys::FaissDistanceComputer;

    fn set_query(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
        rc!({ sys::faiss_DistanceComputer_set_query(self.ptr(), x.as_ref().as_ptr()) })
    }

    fn vector_to_query_dis(&self, i: i64) -> Result<f32> {
        let mut r = 0.0f32;
        rc!({ sys::faiss_DistanceComputer_vector_to_query_dis(self.ptr(), i, &mut r) })?;
        Ok(r)
    }

    fn symmetric_dis(&self, i: i64, j: i64) -> Result<f32> {
        let mut r = 0.0f32;
        rc!({ sys::faiss_DistanceComputer_symmetric_dis(self.ptr(), i, j, &mut r) })?;
        Ok(r)
    }
}

#[cfg(test)]
#[test]
fn test_id_selector_ok() -> Result<()> {
    std::env::set_var("RUST_LOG", "trace");
    let _ = tracing_subscriber::fmt::try_init();
    let sel = IDSelectorBatch::new([1, 2, 3])?
        .not()?
        .or(IDSelectorRange::new(0, 10)?)?;
    assert!(sel.is_member(1));
    Ok(())
}
