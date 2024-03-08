use std::marker::PhantomData;
use std::ptr::null_mut;

use crate::error::{Error, Result};
use crate::rc;
use faiss_next_sys as sys;
use tracing::*;

use super::common::{impl_index_drop, impl_index_trait, FaissIndexTrait};

pub trait FaissIndexIDMapTrait: FaissIndexTrait + Sized {
    fn id_map(&self) -> &[i64];
    fn index(&self) -> FaissIndexIDMapSubIndex<'_, Self>;
}

pub struct FaissIndexIDMap {
    inner: *mut sys::FaissIndexIDMap,
}

impl_index_drop!(FaissIndexIDMap, faiss_Index_free);
impl_index_trait!(FaissIndexIDMap);

impl FaissIndexIDMapTrait for FaissIndexIDMap {
    fn id_map(&self) -> &[i64] {
        let mut len = 0usize;
        let mut data = null_mut();
        unsafe { sys::faiss_IndexIDMap_id_map(self.inner, &mut data, &mut len) };
        unsafe { std::slice::from_raw_parts(data, len) }
    }

    fn index(&self) -> FaissIndexIDMapSubIndex<'_, Self> {
        let inner = unsafe { sys::faiss_IndexIDMap_sub_index(self.inner) };
        FaissIndexIDMapSubIndex {
            inner,
            marker: PhantomData,
        }
    }
}

impl FaissIndexIDMap {
    pub fn new(index: impl FaissIndexTrait) -> Result<Self> {
        let i = index.into_inner();
        let mut inner = std::ptr::null_mut();
        rc!({ sys::faiss_IndexIDMap_new(&mut inner, i) })?;
        trace!("create faiss index inner={:?}, index={:?}", inner, i);
        Ok(Self { inner })
    }

    pub fn downcast(index: impl FaissIndexTrait) -> Result<Self> {
        let i = index.into_inner();
        let i = unsafe { sys::faiss_IndexIDMap_cast(i) };
        match i.is_null() {
            true => {
                error!("Failed to downcast index to FaissIndexIDMap");
                Err(Error::DowncastFailure)
            }
            false => Ok(Self { inner: i }),
        }
    }
}

pub struct FaissIndexIDMapSubIndex<'a, T: FaissIndexIDMapTrait> {
    inner: *mut sys::FaissIndex,
    marker: PhantomData<&'a T>,
}

impl<T> FaissIndexTrait for FaissIndexIDMapSubIndex<'_, T>
where
    T: FaissIndexIDMapTrait,
{
    fn inner(&self) -> *mut sys::FaissIndex {
        self.inner
    }

    fn into_inner(self) -> *mut sys::FaissIndex {
        unimplemented!()
    }
}

pub struct FaissIndexIDMap2 {
    inner: *mut sys::FaissIndexIDMap2,
}

impl_index_drop!(FaissIndexIDMap2, faiss_Index_free);
impl_index_trait!(FaissIndexIDMap2);

impl FaissIndexIDMap2 {
    pub fn new(index: impl FaissIndexTrait) -> Result<Self> {
        let i = index.into_inner();
        let mut inner = std::ptr::null_mut();
        rc!({ sys::faiss_IndexIDMap2_new(&mut inner, i) })?;
        unsafe { sys::faiss_IndexIDMap2_set_own_fields(inner, true as i32) };
        Ok(Self { inner })
    }

    pub fn downcast(index: impl FaissIndexTrait) -> Result<Self> {
        let i = index.into_inner();
        let i = unsafe { sys::faiss_IndexIDMap2_cast(i) };
        match i.is_null() {
            true => {
                error!("Failed to downcast index to FaissIndexIDMap");
                Err(Error::DowncastFailure)
            }
            false => Ok(Self { inner: i }),
        }
    }
}

impl FaissIndexIDMapTrait for FaissIndexIDMap2 {
    fn id_map(&self) -> &[i64] {
        let mut len = 0usize;
        let mut data = null_mut();
        unsafe { sys::faiss_IndexIDMap2_id_map(self.inner, &mut data, &mut len) };
        unsafe { std::slice::from_raw_parts(data, len) }
    }
    fn index(&self) -> FaissIndexIDMapSubIndex<'_, Self> {
        let inner = unsafe { sys::faiss_IndexIDMap2_sub_index(self.inner) };
        FaissIndexIDMapSubIndex {
            inner,
            marker: PhantomData,
        }
    }
}
