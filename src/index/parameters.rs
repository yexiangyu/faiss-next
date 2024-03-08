use std::ptr::null_mut;

use super::id_selector::FaissIDSelectorTrait;
use crate::error::Result;
use faiss_next_sys as sys;

pub trait FaissSearchParametersTrait {
    fn inner(&self) -> *mut sys::FaissSearchParameters;
    fn into_inner(self) -> *mut sys::FaissSearchParameters;
}

pub struct FaissSearchParametersImpl {
    inner: *mut sys::FaissSearchParameters,
}

impl FaissSearchParametersTrait for FaissSearchParametersImpl {
    fn inner(&self) -> *mut sys::FaissSearchParameters {
        self.inner
    }

    fn into_inner(self) -> *mut sys::FaissSearchParameters {
        let mut s = self;
        let inner = s.inner;
        s.inner = null_mut();
        inner
    }
}

impl FaissSearchParametersImpl {
    pub fn null() -> Self {
        Self {
            inner: std::ptr::null_mut(),
        }
    }

    pub fn new(id_selector: impl FaissIDSelectorTrait) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        crate::rc!({ sys::faiss_SearchParameters_new(&mut inner, id_selector.inner()) })?;
        Ok(Self { inner })
    }
}

impl Drop for FaissSearchParametersImpl {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { sys::faiss_SearchParameters_free(self.inner) }
        }
    }
}
