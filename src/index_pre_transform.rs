use crate::{
    error::*,
    index::{FaissIndexBorrowed, FaissIndexTrait},
    macros::*,
    traits::FaissVectorTransformTrait,
};
use faiss_next_sys as ffi;
use std::{marker::PhantomData, mem::forget, ptr::null_mut};

#[derive(Debug)]
pub struct FaissIndexPreTransform {
    pub inner: *mut ffi::FaissIndex,
}

impl_faiss_drop!(FaissIndexPreTransform, faiss_IndexPreTransform_free);
impl FaissIndexTrait for FaissIndexPreTransform {
    fn inner(&self) -> *mut faiss_next_sys::FaissIndex {
        self.inner
    }
}
impl FaissIndexPreTransform {
    pub fn new() -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexPreTransform_new(&mut inner) })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }

    pub fn new_with(index: impl FaissIndexTrait) -> Result<Self> {
        let index_inner = index.inner();
        forget(index);
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexPreTransform_new_with(&mut inner, index_inner) })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }

    pub fn prepend_transform(&mut self, trans: impl FaissVectorTransformTrait) -> Result<()> {
        let trans_inner = trans.inner();
        forget(trans);
        faiss_rc(unsafe { ffi::faiss_IndexPreTransform_prepend_transform(self.inner, trans_inner) })
    }

    pub fn downcast(index: impl FaissIndexTrait) -> Result<Self> {
        let inner = index.inner();
        forget(index);
        let inner = unsafe { ffi::faiss_IndexPreTransform_cast(inner) };
        Ok(Self { inner })
    }

    pub fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexPreTransform_own_fields(self.inner) > 0 }
    }

    fn set_own_fields(&mut self, value: bool) {
        unsafe { ffi::faiss_IndexPreTransform_set_own_fields(self.inner, value as i32) }
    }

    pub fn index(&self) -> Option<FaissIndexBorrowed<'_, Self>> {
        let inner = unsafe { ffi::faiss_IndexPreTransform_index(self.inner) };
        match inner.is_null() {
            true => None,
            false => Some(FaissIndexBorrowed {
                inner,
                owner: PhantomData,
            }),
        }
    }
}
