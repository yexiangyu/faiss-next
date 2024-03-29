use std::{mem::forget, ptr::null_mut};

use faiss_next_sys as sys;

use crate::{
    error::Result,
    index::{impl_index, IndexTrait},
    macros::rc,
};

pub struct IndexIDMap {
    inner: *mut sys::FaissIndexIDMap,
}

impl_index!(IndexIDMap);

impl IndexIDMap {
    pub fn new(index: impl IndexTrait) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexIDMap_new(&mut inner, index.ptr()) })?;
        forget(index);
        Ok(Self { inner })
    }

    pub fn cast(index: impl IndexTrait) -> Self {
        let inner = unsafe { sys::faiss_IndexIDMap_cast(index.ptr()) };
        forget(index);
        Self { inner }
    }

    pub fn id_map(&mut self) -> &mut [i64] {
        let mut p_id_map = null_mut();
        let mut p_size = 0usize;
        unsafe {
            sys::faiss_IndexIDMap_id_map(self.inner, &mut p_id_map, &mut p_size);
            std::slice::from_raw_parts_mut(p_id_map, p_size)
        }
    }
}

pub struct IndexIDMap2 {
    inner: *mut sys::FaissIndexIDMap2,
}

impl_index!(IndexIDMap2);

impl IndexIDMap2 {
    pub fn new(index: impl IndexTrait) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexIDMap_new(&mut inner, index.ptr()) })?;
        forget(index);
        Ok(Self { inner })
    }

    pub fn construct_rev_map(&mut self) -> Result<()> {
        rc!({ sys::faiss_IndexIDMap2_construct_rev_map(self.inner) })
    }

    pub fn cast(index: impl IndexTrait) -> Self {
        let inner = unsafe { sys::faiss_IndexIDMap2_cast(index.ptr()) };
        forget(index);
        Self { inner }
    }

    pub fn id_map(&mut self) -> &mut [i64] {
        let mut p_id_map = null_mut();
        let mut p_size = 0usize;
        unsafe {
            sys::faiss_IndexIDMap_id_map(self.inner, &mut p_id_map, &mut p_size);
            std::slice::from_raw_parts_mut(p_id_map, p_size)
        }
    }
}
