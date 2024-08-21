use itertools::Itertools;
use std::{marker::PhantomData, ptr::null_mut, slice::from_raw_parts};

use crate::{error::*, macros::*};

use faiss_next_sys as ffi;

#[derive(Debug)]
pub struct FaissRangeSearchResult {
    pub inner: *mut ffi::FaissRangeSearchResult,
}

impl_faiss_getter!(
    FaissRangeSearchResult,
    nq,
    faiss_RangeSearchResult_nq,
    usize
);

impl_faiss_new!(
    FaissRangeSearchResult,
    new,
    FaissRangeSearchResult,
    faiss_RangeSearchResult_new,
    nq,
    i64
);

impl_faiss_new!(
    FaissRangeSearchResult,
    new_with,
    FaissRangeSearchResult,
    faiss_RangeSearchResult_new_with,
    nq,
    i64,
    alloc_lims,
    i32
);

impl_faiss_functioin_rc!(
    FaissRangeSearchResult,
    do_allocation,
    faiss_RangeSearchResult_do_allocation
);

impl_faiss_drop!(FaissRangeSearchResult, faiss_RangeSearchResult_free);

impl_faiss_getter!(
    FaissRangeSearchResult,
    buffer_size,
    faiss_RangeSearchResult_buffer_size,
    usize
);

impl_faiss_functioin_void!(
    FaissRangeSearchResult,
    raw_lims,
    faiss_RangeSearchResult_lims,
    lims,
    *mut *mut usize
);

impl_faiss_functioin_void!(
    FaissRangeSearchResult,
    raw_labels,
    faiss_RangeSearchResult_labels,
    labels,
    *mut *mut i64,
    distances,
    *mut *mut f32
);

impl FaissRangeSearchResult {
    pub fn lims(&self) -> &[usize] {
        let mut lims = null_mut();
        self.raw_lims(&mut lims);
        let lims = unsafe { from_raw_parts(lims, self.nq() + 1) };
        lims
    }

    pub fn labels(&self) -> Vec<(&[i64], &[f32])> {
        let mut labels = null_mut();
        let mut distances = null_mut();
        self.raw_labels(&mut labels, &mut distances);
        self.lims()
            .windows(2)
            .map(|indice| {
                let s = indice[0];
                let e = indice[1];
                let labels = unsafe { from_raw_parts(labels.add(s), e - s) };
                let distances = unsafe { from_raw_parts(distances.add(s), e - s) };
                (labels, distances)
            })
            .collect_vec()
    }
}

pub trait FaissIDSelectorTrait {
    fn inner(&self) -> *mut ffi::FaissIDSelector;
    fn is_member(&self, id: i64) -> bool {
        unsafe { ffi::faiss_IDSelector_is_member(self.inner() as *const _, id) > 0 }
    }
}

macro_rules! impl_idselector {
    ($klass: ident) => {
        impl FaissIDSelectorTrait for $klass {
            fn inner(&self) -> *mut ffi::FaissIDSelector {
                self.inner as *mut _
            }
        }
    };
}

pub struct FaissIDSelectorBorrowed<'a, T> {
    pub inner: *const ffi::FaissIDSelector,
    pub owner: PhantomData<&'a T>,
}

impl<'a, T> FaissIDSelectorTrait for FaissIDSelectorBorrowed<'a, T> {
    fn inner(&self) -> *mut ffi::FaissIDSelector {
        self.inner as *mut _
    }
}

///```rust
/// use faiss_next::impl_aux_index_structure::FaissIDSelectorRange;
/// use faiss_next::traits::FaissIDSelectorTrait;
/// let sel = FaissIDSelectorRange::new(1, 10).unwrap();
/// assert!(sel.is_member(5));
/// ```
pub struct FaissIDSelectorRange {
    pub inner: *mut ffi::FaissIDSelectorRange,
}
impl_idselector!(FaissIDSelectorRange);
impl_faiss_drop!(FaissIDSelectorRange, faiss_IDSelectorRange_free);
impl_faiss_getter!(FaissIDSelectorRange, imin, faiss_IDSelectorRange_imin, i64);
impl_faiss_getter!(FaissIDSelectorRange, imax, faiss_IDSelectorRange_imax, i64);
impl_faiss_new!(
    FaissIDSelectorRange,
    new,
    FaissIDSelectorRange,
    faiss_IDSelectorRange_new,
    imin,
    i64,
    imax,
    i64
);

pub struct FaissIDSelectorBatch {
    pub inner: *mut ffi::FaissIDSelectorBatch,
}
impl_idselector!(FaissIDSelectorBatch);
impl_faiss_drop_as!(FaissIDSelectorBatch, faiss_IDSelector_free);
impl_faiss_getter!(
    FaissIDSelectorBatch,
    nbits,
    faiss_IDSelectorBatch_nbits,
    i32
);
impl_faiss_getter!(FaissIDSelectorBatch, mask, faiss_IDSelectorBatch_mask, i64);
impl_faiss_functioin_static_rc!(
    FaissIDSelectorBatch,
    raw_new,
    faiss_IDSelectorBatch_new,
    inner,
    *mut *mut ffi::FaissIDSelectorBatch,
    n,
    usize,
    indices,
    *const i64
);
impl FaissIDSelectorBatch {
    pub fn new(indices: impl AsRef<[i64]>) -> Result<Self> {
        let indices = indices.as_ref();
        let mut inner = null_mut();
        Self::raw_new(&mut inner, indices.len(), indices.as_ptr())?;
        Ok(Self { inner })
    }
}

pub struct FaissIDSelectorNot {
    pub inner: *mut ffi::FaissIDSelectorNot,
}
impl_idselector!(FaissIDSelectorNot);
impl_faiss_functioin_static_rc!(
    FaissIDSelectorNot,
    raw_new,
    faiss_IDSelectorNot_new,
    inner,
    *mut *mut ffi::FaissIDSelectorNot,
    rhs,
    *const ffi::FaissIDSelector
);
impl FaissIDSelectorNot {
    pub fn new(rhs: impl AsRef<dyn FaissIDSelectorTrait>) -> Result<Self> {
        let rhs = rhs.as_ref();
        let mut inner = null_mut();
        Self::raw_new(&mut inner, rhs.inner())?;
        Ok(Self { inner })
    }
}

pub struct FaissIDSelectorAnd {
    pub inner: *mut ffi::FaissIDSelectorAnd,
}
impl_idselector!(FaissIDSelectorAnd);
impl_faiss_functioin_static_rc!(
    FaissIDSelectorAnd,
    raw_new,
    faiss_IDSelectorAnd_new,
    inner,
    *mut *mut ffi::FaissIDSelectorAnd,
    lhs,
    *const ffi::FaissIDSelector,
    rhs,
    *const ffi::FaissIDSelector
);
impl FaissIDSelectorAnd {
    pub fn new(
        lhs: impl AsRef<dyn FaissIDSelectorTrait>,
        rhs: impl AsRef<dyn FaissIDSelectorTrait>,
    ) -> Result<Self> {
        let rhs = rhs.as_ref();
        let lhs = lhs.as_ref();
        let mut inner = null_mut();
        Self::raw_new(&mut inner, lhs.inner(), rhs.inner())?;
        Ok(Self { inner })
    }
}

pub struct FaissIDSelectorOr {
    pub inner: *mut ffi::FaissIDSelectorOr,
}
impl_idselector!(FaissIDSelectorOr);
impl_faiss_functioin_static_rc!(
    FaissIDSelectorOr,
    raw_new,
    faiss_IDSelectorOr_new,
    inner,
    *mut *mut ffi::FaissIDSelectorOr,
    lhs,
    *const ffi::FaissIDSelector,
    rhs,
    *const ffi::FaissIDSelector
);
impl FaissIDSelectorOr {
    pub fn new(
        lhs: impl AsRef<dyn FaissIDSelectorTrait>,
        rhs: impl AsRef<dyn FaissIDSelectorTrait>,
    ) -> Result<Self> {
        let rhs = rhs.as_ref();
        let lhs = lhs.as_ref();
        let mut inner = null_mut();
        Self::raw_new(&mut inner, lhs.inner(), rhs.inner())?;
        Ok(Self { inner })
    }
}

pub struct FaissIDSelectorXOr {
    pub inner: *mut ffi::FaissIDSelectorXOr,
}
impl_idselector!(FaissIDSelectorXOr);
impl_faiss_functioin_static_rc!(
    FaissIDSelectorXOr,
    raw_new,
    faiss_IDSelectorXOr_new,
    inner,
    *mut *mut ffi::FaissIDSelectorXOr,
    lhs,
    *const ffi::FaissIDSelector,
    rhs,
    *const ffi::FaissIDSelector
);
impl FaissIDSelectorXOr {
    pub fn new(
        lhs: impl AsRef<dyn FaissIDSelectorTrait>,
        rhs: impl AsRef<dyn FaissIDSelectorTrait>,
    ) -> Result<Self> {
        let rhs = rhs.as_ref();
        let lhs = lhs.as_ref();
        let mut inner = null_mut();
        Self::raw_new(&mut inner, lhs.inner(), rhs.inner())?;
        Ok(Self { inner })
    }
}

pub struct FaissBufferList {
    inner: *mut ffi::FaissBufferList,
}
impl_faiss_drop!(FaissBufferList, faiss_BufferList_free);
impl_faiss_getter!(
    FaissBufferList,
    buffer_size,
    faiss_BufferList_buffer_size,
    usize
);
impl_faiss_getter!(FaissBufferList, wp, faiss_BufferList_wp, usize);
impl_faiss_functioin_rc!(
    FaissBufferList,
    append_buffer,
    faiss_BufferList_append_buffer
);
impl_faiss_new!(
    FaissBufferList,
    new,
    FaissBufferList,
    faiss_BufferList_new,
    buffer_size,
    usize
);
impl_faiss_functioin_rc!(
    FaissBufferList,
    add,
    faiss_BufferList_add,
    id,
    i64,
    dis,
    f32
);
impl_faiss_functioin_rc!(
    FaissBufferList,
    raw_copy_range,
    faiss_BufferList_copy_range,
    ofs,
    usize,
    n,
    usize,
    dest_ids,
    *mut i64,
    dest_dis,
    *mut f32
);

// ffi::FaissRangeSearchPartialResult no new or free function
// ffi::FaissRangeQueryResult no new or free function
// ffi::DistanceComputer no new or free function
