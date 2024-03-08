use std::{marker::PhantomData, ptr::null_mut};

use crate::error::Result;
use crate::rc;
use faiss_next_sys as sys;

use super::common::{impl_index_drop, impl_index_trait, FaissIndexTrait};

pub struct FaissIndexShards {
    inner: *mut sys::FaissIndexShards,
}

impl_index_drop!(FaissIndexShards, faiss_IndexShards_free);
impl_index_trait!(FaissIndexShards);

pub struct FaissIndexShardsIndex<'a> {
    inner: *mut sys::FaissIndex,
    marker: PhantomData<&'a FaissIndexShards>,
}

impl FaissIndexTrait for FaissIndexShardsIndex<'_> {
    fn inner(&self) -> *mut sys::FaissIndex {
        self.inner
    }

    fn into_inner(self) -> *mut sys::FaissIndex {
        unimplemented!()
    }
}

impl FaissIndexShards {
    pub fn own_fields(&self) -> bool {
        unsafe { sys::faiss_IndexShards_own_fields(self.inner) != 0 }
    }

    pub fn set_own_fields(&mut self, value: bool) {
        unsafe { sys::faiss_IndexShards_set_own_fields(self.inner, value as i32) }
    }

    pub fn sucessive_ids(&self) -> bool {
        unsafe { sys::faiss_IndexShards_successive_ids(self.inner) != 0 }
    }

    pub fn new(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexShards_new(&mut inner, d) })?;
        Ok(Self { inner })
    }

    pub fn new_with_options(d: i64, threaded: bool, successive_ids: bool) -> Result<Self> {
        let mut inner = null_mut();
        rc!({
            sys::faiss_IndexShards_new_with_options(
                &mut inner,
                d,
                threaded as i32,
                successive_ids as i32,
            )
        })?;
        Ok(Self { inner })
    }

    pub fn add_shard(&mut self, index: impl FaissIndexTrait) -> Result<()> {
        let i = index.into_inner();
        rc!({ sys::faiss_IndexShards_add_shard(self.inner, i) })?;
        Ok(())
    }

    pub fn remove_shard(&mut self, index: impl FaissIndexTrait) -> Result<()> {
        let i = index.into_inner();
        rc!({ sys::faiss_IndexShards_remove_shard(self.inner, i) })?;
        Ok(())
    }

    pub fn at(&mut self, i: i32) -> FaissIndexShardsIndex {
        let inner = unsafe { sys::faiss_IndexShards_at(self.inner, i) };
        FaissIndexShardsIndex {
            inner,
            marker: PhantomData,
        }
    }
}
