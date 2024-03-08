use crate::error::Result;
use crate::rc;
use std::ptr::{addr_of_mut, null_mut};

use faiss_next_sys as sys;
use tracing::trace;

pub trait FaissIDSelectorTrait {
    fn inner(&self) -> *mut sys::FaissIDSelector;

    fn is_member(&self, id: i64) -> bool {
        unsafe { sys::faiss_IDSelector_is_member(self.inner(), id) != 0 }
    }

    fn not(self) -> Result<FaissIDSelectorNot>
    where
        Self: Sized + 'static,
    {
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorNot_new(addr_of_mut!(inner), self.inner()) })?;
        trace!("create id_selector not, inner={:?}", inner);
        Ok(FaissIDSelectorNot {
            inner: inner as *mut _,
            source: Box::new(self),
        })
    }

    fn and(self, rhs: impl FaissIDSelectorTrait + 'static) -> Result<FaissIDSelectorAnd>
    where
        Self: Sized + 'static,
    {
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorAnd_new(addr_of_mut!(inner), self.inner(), rhs.inner()) })?;
        trace!("create id_selector and, inner={:?}", inner);
        Ok(FaissIDSelectorAnd {
            inner: inner as *mut _,
            l: Box::new(self),
            r: Box::new(rhs),
        })
    }

    fn or(self, rhs: impl FaissIDSelectorTrait + 'static) -> Result<FaissIDSelectorOr>
    where
        Self: Sized + 'static,
    {
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorOr_new(addr_of_mut!(inner), self.inner(), rhs.inner()) })?;
        trace!("create id_selector or, inner={:?}", inner);
        Ok(FaissIDSelectorOr {
            inner: inner as *mut _,
            l: Box::new(self),
            r: Box::new(rhs),
        })
    }
}

macro_rules! impl_id_selector_common {
    ($kls: ident) => {
        impl FaissIDSelectorTrait for $kls {
            fn inner(&self) -> *mut sys::FaissIDSelector {
                self.inner as *mut _
            }
        }

        impl Drop for $kls {
            fn drop(&mut self) {
                unsafe { sys::faiss_IDSelector_free(self.inner()) }
                trace!("drop id_selector={:?}", self.inner);
            }
        }
    };
}

pub struct FaissIDSelectorBatch {
    pub inner: *mut sys::FaissIDSelectorBatch,
}

impl_id_selector_common!(FaissIDSelectorBatch);

impl FaissIDSelectorBatch {
    pub fn new(&self, ids: &[i64]) -> Result<Self> {
        let n = ids.len();
        let ids = ids.as_ptr();
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorBatch_new(addr_of_mut!(inner), n, ids) })?;
        trace!("create id_selector batch, inner={:?}", inner);
        Ok(Self {
            inner: inner as *mut _,
        })
    }

    pub fn nbits(&self) -> i32 {
        unsafe { sys::faiss_IDSelectorBatch_nbits(self.inner) }
    }

    pub fn mask(&self) -> i64 {
        unsafe { sys::faiss_IDSelectorBatch_mask(self.inner) }
    }
}

pub struct FaissIDSelectorRange {
    pub inner: *mut sys::FaissIDSelectorRange,
}

impl FaissIDSelectorTrait for FaissIDSelectorRange {
    fn inner(&self) -> *mut sys::FaissIDSelector {
        self.inner as *mut _
    }
}

impl Drop for FaissIDSelectorRange {
    fn drop(&mut self) {
        unsafe { sys::faiss_IDSelectorRange_free(self.inner) }
        trace!("drop range id_selector={:?}", self.inner);
    }
}

impl FaissIDSelectorRange {
    pub fn new(min: i64, max: i64) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorRange_new(addr_of_mut!(inner), min, max) })?;
        trace!("create id_selector range, inner={:?}", inner);
        Ok(Self {
            inner: inner as *mut _,
        })
    }

    pub fn min(&self) -> i64 {
        unsafe { sys::faiss_IDSelectorRange_imin(self.inner) }
    }

    pub fn max(&self) -> i64 {
        unsafe { sys::faiss_IDSelectorRange_imax(self.inner) }
    }
}

pub struct FaissIDSelectorNot {
    inner: *mut sys::FaissIDSelectorNot,
    #[allow(unused)]
    source: Box<dyn FaissIDSelectorTrait>,
}

impl_id_selector_common!(FaissIDSelectorNot);

pub struct FaissIDSelectorAnd {
    inner: *mut sys::FaissIDSelectorAnd,
    #[allow(unused)]
    l: Box<dyn FaissIDSelectorTrait>,
    #[allow(unused)]
    r: Box<dyn FaissIDSelectorTrait>,
}

impl_id_selector_common!(FaissIDSelectorAnd);

pub struct FaissIDSelectorOr {
    inner: *mut sys::FaissIDSelectorOr,
    #[allow(unused)]
    l: Box<dyn FaissIDSelectorTrait>,
    #[allow(unused)]
    r: Box<dyn FaissIDSelectorTrait>,
}

impl_id_selector_common!(FaissIDSelectorOr);

pub struct FaissIDSelectorXOr {
    inner: *mut sys::FaissIDSelectorXOr,
    #[allow(unused)]
    l: Box<dyn FaissIDSelectorTrait>,
    #[allow(unused)]
    r: Box<dyn FaissIDSelectorTrait>,
}

impl_id_selector_common!(FaissIDSelectorXOr);
