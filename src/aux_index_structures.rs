use std::ptr::null_mut;

use faiss_next_sys as sys;

use crate::{error::Result, macros::rc};

pub struct RangeSearchResult {
    inner: *mut sys::FaissRangeSearchResult,
}

impl Drop for RangeSearchResult {
    fn drop(&mut self) {
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
        Ok(Self { inner })
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

impl IDSelectorRange {
    pub fn new(imin: i64, imax: i64) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorRange_new(&mut inner, imin, imax) })?;
        Ok(Self { inner })
    }
}

pub struct IDSelectorBatch {
    inner: *mut sys::FaissIDSelectorBatch,
}
impl_drop!(IDSelectorBatch, faiss_IDSelector_free);
impl_id_selector!(IDSelectorBatch);

impl IDSelectorBatch {
    pub fn new(ids: impl AsRef<[i64]>) -> Result<Self> {
        let mut inner = null_mut();
        let ids = ids.as_ref();
        let n = ids.len();
        rc!({ sys::faiss_IDSelectorBatch_new(&mut inner, n, ids.as_ptr()) })?;
        Ok(Self { inner })
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
