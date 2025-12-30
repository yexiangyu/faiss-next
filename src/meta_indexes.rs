use crate::error::Result;
use faiss_next_sys as ffi;

use std::{
    ptr::null_mut,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use crate::index::IndexTrait;

pub trait IndexIDMapTrait: IndexTrait {
    fn own_fields(&self) -> bool {
        ffi::run!(
            faiss_IndexIDMap_own_fields,
            self.inner() as *mut ffi::FaissIndexIDMap
        ) != 0
    }

    fn id_map(&self) -> &[i64] {
        let mut size = 0;
        let mut ptr = null_mut();
        ffi::run!(faiss_IndexIDMap_id_map, self.inner(), &mut ptr, &mut size);
        unsafe { from_raw_parts(ptr, size) }
    }

    fn id_map_mut(&mut self) -> &mut [i64] {
        let mut size = 0;
        let mut ptr = null_mut();
        ffi::run!(faiss_IndexIDMap_id_map, self.inner(), &mut ptr, &mut size);
        unsafe { from_raw_parts_mut(ptr, size) }
    }

    fn sub_index(&self) -> crate::index::IndexBorrowed<'_> {
        let inner = ffi::run!(faiss_IndexIDMap_sub_index, self.inner());
        crate::index::IndexBorrowed::new(inner)
    }
}

macro_rules! impl_index_id_map {
    ($cls: ident) => {
        impl IndexTrait for $cls {
            fn inner(&self) -> *mut ffi::FaissIndex {
                self.inner
            }
        }

        impl IndexIDMapTrait for $cls {}
    };
}

#[derive(Debug)]
pub struct IndexIDMap {
    inner: *mut ffi::FaissIndex,
}

impl_index_id_map!(IndexIDMap);
ffi::impl_drop!(IndexIDMap, faiss_Index_free);

impl IndexIDMap {
    pub fn new(index: impl IndexTrait) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexIDMap_new, &mut inner, index.inner())?;
        ffi::run!(faiss_IndexIDMap_set_own_fields, inner, true as i32);
        Ok(Self { inner })
    }

    pub fn cast(index: impl IndexTrait) -> Self {
        let inner = ffi::run!(faiss_IndexIDMap_cast, index.inner());
        assert!(!inner.is_null(), "Failed to cast index");
        Self { inner }
    }
}

pub trait IndexIDMap2Trait: IndexTrait {
    fn id_map(&self) -> &[i64] {
        let mut size = 0;
        let mut ptr = null_mut();
        unsafe {
            ffi::faiss_IndexIDMap2_id_map(
                self.inner() as *mut ffi::FaissIndexIDMap2,
                &mut ptr,
                &mut size,
            )
        };
        unsafe { from_raw_parts(ptr, size) }
    }

    fn id_map_mut(&mut self) -> &mut [i64] {
        let mut size = 0;
        let mut ptr = null_mut();
        unsafe {
            ffi::faiss_IndexIDMap2_id_map(
                self.inner() as *mut ffi::FaissIndexIDMap2,
                &mut ptr,
                &mut size,
            )
        };
        unsafe { from_raw_parts_mut(ptr, size) }
    }

    fn sub_index(&self) -> crate::index::IndexBorrowed<'_> {
        let inner =
            unsafe { ffi::faiss_IndexIDMap2_sub_index(self.inner() as *mut ffi::FaissIndexIDMap2) };
        crate::index::IndexBorrowed::new(inner)
    }

    fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexIDMap2_own_fields(self.inner() as *mut ffi::FaissIndexIDMap2) > 0 }
    }

    fn construct_rev_map(&mut self) -> Result<()> {
        let result = unsafe {
            ffi::faiss_IndexIDMap2_construct_rev_map(self.inner() as *mut ffi::FaissIndexIDMap2)
        };
        ffi::rc(result).map_err(crate::error::Error::from)
    }
}

macro_rules! impl_index_id_map2 {
    ($cls: ident) => {
        impl IndexTrait for $cls {
            fn inner(&self) -> *mut ffi::FaissIndex {
                self.inner
            }
        }

        impl IndexIDMap2Trait for $cls {}
    };
}

#[derive(Debug)]
pub struct IndexIDMap2 {
    inner: *mut ffi::FaissIndex,
}

impl_index_id_map2!(IndexIDMap2);
ffi::impl_drop!(IndexIDMap2, faiss_Index_free);

impl IndexIDMap2 {
    pub fn new(index: impl IndexTrait) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexIDMap2_new, &mut inner, index.inner())?;
        ffi::run!(faiss_IndexIDMap2_set_own_fields, inner, true as i32);
        Ok(Self { inner })
    }

    pub fn cast(index: impl IndexTrait) -> Result<Self> {
        let inner = ffi::run!(faiss_IndexIDMap2_cast, index.inner());
        if inner.is_null() {
            return Err(crate::error::Error::Faiss(ffi::Error {
                code: -1, // Use -1 as a generic error code for failed cast
                message: "Failed to cast index to IndexIDMap2".to_string(),
            }));
        }
        Ok(Self { inner })
    }
}
