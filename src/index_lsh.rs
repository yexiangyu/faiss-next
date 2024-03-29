use std::mem::forget;

use faiss_next_sys as sys;

use crate::index::{impl_index, IndexTrait};

pub struct IndexLSH {
    inner: *mut sys::FaissIndexLSH,
}
impl_index!(IndexLSH);

impl IndexLSH {
    pub fn new(d: i64, nbits: i32, rotate_data: bool, train_threshold: bool) -> Self {
        let mut inner = std::ptr::null_mut();
        unsafe {
            sys::faiss_IndexLSH_new_with_options(
                &mut inner,
                d,
                nbits,
                rotate_data as i32,
                train_threshold as i32,
            )
        };
        Self { inner }
    }

    pub fn cast(index: impl IndexTrait) -> Self {
        let ptr = index.ptr();
        forget(index);
        let inner = unsafe { sys::faiss_IndexLSH_cast(ptr) };
        Self { inner }
    }

    pub fn nbits(&self) -> i32 {
        unsafe { sys::faiss_IndexLSH_nbits(self.inner) }
    }

    pub fn code_size(&self) -> i32 {
        unsafe { sys::faiss_IndexLSH_code_size(self.inner) }
    }

    pub fn rotate_data(&self) -> bool {
        unsafe { sys::faiss_IndexLSH_rotate_data(self.inner) != 0 }
    }

    pub fn train_thresholds(&self) -> bool {
        unsafe { sys::faiss_IndexLSH_train_thresholds(self.inner) != 0 }
    }
}
