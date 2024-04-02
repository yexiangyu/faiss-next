use std::{mem::forget, ptr::null_mut};

use faiss_next_sys as sys;

use crate::{
    error::Result,
    index::{impl_index, IndexTrait},
    macros::rc,
};

pub struct IndexShards {
    inner: *mut sys::FaissIndexShards,
    num_shards: usize,
}

impl_index!(IndexShards);

impl IndexShards {
    pub fn new(d: i64, threaded: bool, succesive_ids: bool) -> Result<Self> {
        let mut inner = null_mut();
        rc!({
            sys::faiss_IndexShards_new_with_options(
                &mut inner,
                d,
                threaded as i32,
                succesive_ids as i32,
            )
        })?;
        Ok(Self {
            inner,
            num_shards: 0,
        })
    }

    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    pub fn set_own_fields(&mut self, own: bool) {
        unsafe { sys::faiss_IndexShards_set_own_fields(self.inner, own as i32) }
    }

    pub fn add_shard(&mut self, index: impl IndexTrait) -> Result<()> {
        rc!({ sys::faiss_IndexShards_add_shard(self.inner, index.ptr()) })?;
        forget(index);
        self.set_own_fields(true);
        self.num_shards += 1;
        Ok(())
    }

    pub fn remove_shard(&mut self, i: usize) -> Result<()> {
        let r = unsafe { sys::faiss_IndexShards_at(self.ptr(), i as i32) };
        rc!({ sys::faiss_IndexShards_remove_shard(self.inner, r) })?;
        self.num_shards -= 1;
        Ok(())
    }
}
