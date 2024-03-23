use faiss_next_sys as sys;

use crate::index::IndexTrait;
pub struct IndexImpl {
    inner: *mut sys::FaissIndex,
}

impl IndexTrait for IndexImpl {
    fn ptr(&self) -> *mut sys::FaissIndex {
        self.inner
    }
}

impl Drop for IndexImpl {
    fn drop(&mut self) {
        unsafe { sys::faiss_Index_free(self.inner) }
    }
}
