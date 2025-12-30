use crate::error::Result;
use faiss_next_sys as ffi;

use std::ptr::null_mut;

use crate::index::{IndexTrait, IndexBorrowed};

pub trait IndexShardsTrait: IndexTrait {
    fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexShards_own_fields(self.inner()) > 0 }
    }

    fn successive_ids(&self) -> i32 {
        unsafe { ffi::faiss_IndexShards_successive_ids(self.inner()) }
    }

    fn set_own_fields(&mut self, value: bool) {
        unsafe { ffi::faiss_IndexShards_set_own_fields(self.inner(), value as i32) }
    }

    fn add_shard(&mut self, index: impl IndexTrait) -> Result<()> {
        ffi::ok!(faiss_IndexShards_add_shard, self.inner(), index.inner())?;
        Ok(())
    }

    fn remove_shard(&mut self, index: &impl IndexTrait) -> Result<()> {
        ffi::ok!(faiss_IndexShards_remove_shard, self.inner(), index.inner())?;
        Ok(())
    }

    fn at(&self, index: usize) -> Option<IndexBorrowed<'_>> {
        let inner = unsafe { ffi::faiss_IndexShards_at(self.inner(), index as i32) };
        match inner.is_null() {
            true => None,
            false => Some(IndexBorrowed::new(inner)),
        }
    }
}

macro_rules! impl_index_shards {
    ($cls: ident) => {
        impl IndexTrait for $cls {
            fn inner(&self) -> *mut ffi::FaissIndex {
                self.inner
            }
        }

        impl IndexShardsTrait for $cls {}
    };
}

#[derive(Debug)]
pub struct IndexShards {
    inner: *mut ffi::FaissIndex,
}

impl_index_shards!(IndexShards);
ffi::impl_drop!(IndexShards, faiss_IndexShards_free);

impl IndexShards {
    pub fn new(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexShards_new, &mut inner, d)?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }

    pub fn new_with_options(d: i64, threaded: bool, successive_ids: i32) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexShards_new_with_options, &mut inner, d, threaded as i32, successive_ids)?;
        let mut ret = Self { inner };
        ret.set_own_fields(true);
        Ok(ret)
    }
}
