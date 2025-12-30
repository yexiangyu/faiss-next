use crate::error::Result;
use faiss_next_sys as ffi;
use std::{ptr::null_mut, slice::from_raw_parts};

#[derive(Debug)]
pub struct RangeSearchResult {
    pub inner: *mut ffi::FaissRangeSearchResult,
}

impl RangeSearchResult {
    pub fn new(nq: i64) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_RangeSearchResult_new, &mut inner, nq)?;
        Ok(Self { inner })
    }

    pub fn new_with(nq: i64, alloc_lims: bool) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(
            faiss_RangeSearchResult_new_with,
            &mut inner,
            nq,
            alloc_lims as i32
        )?;
        Ok(Self { inner })
    }

    pub fn do_allocation(&self) -> Result<()> {
        ffi::ok!(faiss_RangeSearchResult_do_allocation, self.inner)?;
        Ok(())
    }

    pub fn lims(&self) -> &[usize] {
        let len = self.nq() + 1;
        let mut lims = null_mut();
        ffi::run!(faiss_RangeSearchResult_lims, self.inner, &mut lims);
        unsafe { from_raw_parts(lims, len) }
    }

    pub fn labels(&self) -> Vec<(&[i64], &[f32])> {
        let mut labels = null_mut();
        let mut distances = null_mut();

        ffi::run!(
            faiss_RangeSearchResult_labels,
            self.inner,
            &mut labels,
            &mut distances
        );
        self.lims()
            .windows(2)
            .map(|indice| {
                let s = indice[0];
                let e = indice[1];
                let labels = unsafe { from_raw_parts(labels.add(s), e - s) };
                let distances = unsafe { from_raw_parts(distances.add(s), e - s) };
                (labels, distances)
            })
            .collect()
    }
}

ffi::impl_drop!(RangeSearchResult, faiss_RangeSearchResult_free);
ffi::impl_getter!(RangeSearchResult, nq, faiss_RangeSearchResult_nq, usize);
#[rustfmt::skip]
ffi::impl_getter!(RangeSearchResult, buffer_size, faiss_RangeSearchResult_buffer_size, usize);

pub trait IDSelectorTrait: Sized {
    fn inner(&self) -> *mut ffi::FaissIDSelector;
    fn is_member(&self, id: i64) -> bool {
        ffi::run!(faiss_IDSelector_is_member, self.inner(), id) > 0
    }

    fn and(self, other: impl IDSelectorTrait) -> Result<IDSelectorAnd> {
        IDSelectorAnd::new(self, other)
    }

    fn not(self) -> Result<IDSelectorNot> {
        IDSelectorNot::new(self)
    }

    fn or(self, other: impl IDSelectorTrait) -> Result<IDSelectorOr> {
        IDSelectorOr::new(self, other)
    }

    fn xor(self, other: impl IDSelectorTrait) -> Result<IDSelectorXOr> {
        IDSelectorXOr::new(self, other)
    }
}

macro_rules! impl_id_selector {
    ($cls: ident) => {
        impl IDSelectorTrait for $cls {
            fn inner(&self) -> *mut ffi::FaissIDSelector {
                self.inner as *mut _
            }
        }

        ffi::impl_drop!($cls, faiss_IDSelector_free);
    };
}

#[derive(Debug)]
pub struct IDSelectorRange {
    inner: *mut ffi::FaissIDSelectorRange,
}

impl_id_selector!(IDSelectorRange);
ffi::impl_getter!(IDSelectorRange, imin, faiss_IDSelectorRange_imin, i64);
ffi::impl_getter!(IDSelectorRange, imax, faiss_IDSelectorRange_imax, i64);

impl IDSelectorRange {
    pub fn new(imin: i64, imax: i64) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IDSelectorRange_new, &mut inner, imin, imax)?;
        Ok(Self { inner })
    }
}

#[derive(Debug)]
pub struct IDSelectorBatch {
    inner: *mut ffi::FaissIDSelectorBatch,
}

impl_id_selector!(IDSelectorBatch);
ffi::impl_getter!(IDSelectorBatch, nbits, faiss_IDSelectorBatch_nbits, i32);
ffi::impl_getter!(IDSelectorBatch, mask, faiss_IDSelectorBatch_mask, i64);

impl IDSelectorBatch {
    pub fn new(ids: impl AsRef<[i64]>) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(
            faiss_IDSelectorBatch_new,
            &mut inner,
            ids.as_ref().len(),
            ids.as_ref().as_ptr()
        )?;
        Ok(Self { inner })
    }
}

#[derive(Debug)]
pub struct IDSelectorNot {
    inner: *mut ffi::FaissIDSelectorNot,
}
impl_id_selector!(IDSelectorNot);

impl IDSelectorNot {
    pub fn new(rhs: impl IDSelectorTrait) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IDSelectorNot_new, &mut inner, rhs.inner())?;
        Ok(Self { inner })
    }
}

#[derive(Debug)]
pub struct IDSelectorAnd {
    inner: *mut ffi::FaissIDSelectorAnd,
}
impl_id_selector!(IDSelectorAnd);

impl IDSelectorAnd {
    pub fn new(lhs: impl IDSelectorTrait, rhs: impl IDSelectorTrait) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(
            faiss_IDSelectorAnd_new,
            &mut inner,
            lhs.inner(),
            rhs.inner()
        )?;
        Ok(Self { inner })
    }
}

#[derive(Debug)]
pub struct IDSelectorOr {
    inner: *mut ffi::FaissIDSelectorOr,
}

impl_id_selector!(IDSelectorOr);
impl IDSelectorOr {
    pub fn new(lhs: impl IDSelectorTrait, rhs: impl IDSelectorTrait) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IDSelectorOr_new, &mut inner, lhs.inner(), rhs.inner())?;
        Ok(Self { inner })
    }
}
#[derive(Debug)]
pub struct IDSelectorXOr {
    inner: *mut ffi::FaissIDSelectorXOr,
}
impl_id_selector!(IDSelectorXOr);
impl IDSelectorXOr {
    pub fn new(lhs: impl IDSelectorTrait, rhs: impl IDSelectorTrait) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(
            faiss_IDSelectorXOr_new,
            &mut inner,
            lhs.inner(),
            rhs.inner()
        )?;
        Ok(Self { inner })
    }
}

#[derive(Debug)]
pub struct BufferList {
    inner: *mut ffi::FaissBufferList,
}
ffi::impl_drop!(BufferList, faiss_BufferList_free);
ffi::impl_getter!(BufferList, wp, faiss_BufferList_wp, usize);
ffi::impl_getter!(BufferList, buffer_size, faiss_BufferList_buffer_size, usize);

impl BufferList {
    pub fn new(buffer_size: usize) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_BufferList_new, &mut inner, buffer_size)?;
        Ok(Self { inner })
    }

    pub fn append_buffer(&mut self) -> Result<()> {
        ffi::ok!(faiss_BufferList_append_buffer, self.inner)?;
        Ok(())
    }

    pub fn add(&mut self, id: i64, dis: f32) -> Result<()> {
        ffi::ok!(faiss_BufferList_add, self.inner, id, dis)?;
        Ok(())
    }

    pub fn copy_range(
        &self,
        ofs: usize,
        mut dest_ids: impl AsMut<[i64]>,
        mut dest_dis: impl AsMut<[f32]>,
    ) -> Result<()> {
        let dest_ids = dest_ids.as_mut();
        let dest_dis = dest_dis.as_mut();

        let n = dest_ids.len();

        assert!(
            n == dest_dis.len(),
            "dest_ids and dest_dis must have the same length"
        );

        ffi::ok!(
            faiss_BufferList_copy_range,
            self.inner,
            ofs,
            n,
            dest_ids.as_mut_ptr(),
            dest_dis.as_mut_ptr()
        )?;
        Ok(())
    }
}

// ffi::FaissRangeSearchPartialResult no new or free function
// ffi::FaissRangeQueryResult no new or free function
// ffi::DistanceComputer no new or free function
