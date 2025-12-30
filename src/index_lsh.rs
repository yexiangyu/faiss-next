use crate::error::Result;
use faiss_next_sys as ffi;

use std::ptr::null_mut;

use crate::index::IndexTrait;

pub trait IndexLSHTrait: IndexTrait {
    fn nbits(&self) -> i32 {
        ffi::run!(faiss_IndexLSH_nbits, self.inner())
    }

    fn code_size(&self) -> i32 {
        ffi::run!(faiss_IndexLSH_code_size, self.inner())
    }

    fn rotate_data(&self) -> i32 {
        ffi::run!(faiss_IndexLSH_rotate_data, self.inner())
    }

    fn train_thresholds(&self) -> i32 {
        ffi::run!(faiss_IndexLSH_train_thresholds, self.inner())
    }
}

macro_rules! impl_index_lsh {
    ($cls: ident) => {
        impl IndexTrait for $cls {
            fn inner(&self) -> *mut ffi::FaissIndex {
                self.inner
            }
        }

        impl IndexLSHTrait for $cls {}
    };
}

#[derive(Debug)]
pub struct IndexLSH {
    inner: *mut ffi::FaissIndex,
}

impl_index_lsh!(IndexLSH);
ffi::impl_drop!(IndexLSH, faiss_IndexLSH_free);

impl IndexLSH {
    pub fn new(d: i64, nbits: i32) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexLSH_new, &mut inner, d, nbits)?;
        Ok(Self { inner })
    }

    pub fn new_with_options(
        d: i64,
        nbits: i32,
        rotate_data: bool,
        train_thresholds: bool,
    ) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(
            faiss_IndexLSH_new_with_options,
            &mut inner,
            d,
            nbits,
            rotate_data as i32,
            train_thresholds as i32
        )?;
        Ok(Self { inner })
    }

    pub fn cast(index: impl IndexTrait) -> Result<Self> {
        let inner = ffi::run!(faiss_IndexLSH_cast, index.inner());
        assert!(!inner.is_null(), "Failed to cast index to IndexLSH");
        Ok(Self { inner })
    }
}
