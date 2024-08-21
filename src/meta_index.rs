use std::mem::forget;
use std::ptr::null_mut;

use faiss_next_sys::{self as ffi};

use crate::{
    error::*,
    index::{FaissIndexBorrowed, FaissIndexTrait},
    macros::*,
};

pub struct FaissIndexIDMap {
    inner: *mut ffi::FaissIndexIDMap,
}

impl FaissIndexTrait for FaissIndexIDMap {
    fn inner(&self) -> *mut ffi::FaissIndex {
        self.inner as *mut _
    }
}

impl_faiss_drop!(FaissIndexIDMap, faiss_Index_free);

impl FaissIndexIDMap {
    pub fn new(rhs: impl FaissIndexTrait) -> Result<Self> {
        let rhs_inner = rhs.inner() as *mut _;
        forget(rhs);
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexIDMap_new(&mut inner, rhs_inner) })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }
    pub fn downcast(rhs: impl FaissIndexTrait) -> Self {
        let inner = rhs.inner() as *mut _;
        let inner = unsafe { ffi::faiss_IndexIDMap_cast(inner) };
        forget(rhs);
        Self { inner }
    }

    pub fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexIDMap_own_fields(self.inner) > 0 }
    }

    pub fn id_map(&self) -> &[i64] {
        let mut size = 0;
        let mut ptr = null_mut();
        unsafe { ffi::faiss_IndexIDMap_id_map(self.inner, &mut ptr, &mut size) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    pub fn id_map_mut(&mut self) -> &mut [i64] {
        let mut size = 0;
        let mut ptr = null_mut();
        unsafe { ffi::faiss_IndexIDMap_id_map(self.inner, &mut ptr, &mut size) };
        unsafe { std::slice::from_raw_parts_mut(ptr, size) }
    }

    pub fn sub_index(&self) -> FaissIndexBorrowed<'_, Self> {
        let inner = unsafe { ffi::faiss_IndexIDMap_sub_index(self.inner) };
        FaissIndexBorrowed {
            inner,
            owner: std::marker::PhantomData,
        }
    }

    fn set_own_fields(&mut self, own: bool) {
        unsafe { ffi::faiss_IndexIDMap_set_own_fields(self.inner, own as i32) }
    }
}

pub struct FaissIndexIDMap2 {
    inner: *mut ffi::FaissIndexIDMap2,
}
impl_faiss_drop!(FaissIndexIDMap2, faiss_Index_free);

impl FaissIndexIDMap2 {
    pub fn new(rhs: impl FaissIndexTrait) -> Result<Self> {
        let rhs_inner = rhs.inner() as *mut _;
        forget(rhs);
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_IndexIDMap2_new(&mut inner, rhs_inner) })?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }
    pub fn downcast(rhs: impl FaissIndexTrait) -> Self {
        let inner = rhs.inner() as *mut _;
        let inner = unsafe { ffi::faiss_IndexIDMap2_cast(inner) };
        forget(rhs);
        Self { inner }
    }

    pub fn id_map(&self) -> &[i64] {
        let mut size = 0;
        let mut ptr = null_mut();
        unsafe { ffi::faiss_IndexIDMap2_id_map(self.inner, &mut ptr, &mut size) };
        unsafe { std::slice::from_raw_parts(ptr, size) }
    }

    pub fn id_map_mut(&mut self) -> &mut [i64] {
        let mut size = 0;
        let mut ptr = null_mut();
        unsafe { ffi::faiss_IndexIDMap2_id_map(self.inner, &mut ptr, &mut size) };
        unsafe { std::slice::from_raw_parts_mut(ptr, size) }
    }

    pub fn sub_index(&self) -> FaissIndexBorrowed<'_, Self> {
        let inner = unsafe { ffi::faiss_IndexIDMap2_sub_index(self.inner) };
        FaissIndexBorrowed {
            inner,
            owner: std::marker::PhantomData,
        }
    }

    pub fn construct_rev_map(&mut self) -> Result<()> {
        faiss_rc(unsafe { ffi::faiss_IndexIDMap2_construct_rev_map(self.inner) })
    }

    pub fn own_fiels(&self) -> bool {
        unsafe { ffi::faiss_IndexIDMap2_own_fields(self.inner) > 0 }
    }

    fn set_own_fields(&mut self, own: bool) {
        unsafe { ffi::faiss_IndexIDMap2_set_own_fields(self.inner, own as i32) }
    }
}
