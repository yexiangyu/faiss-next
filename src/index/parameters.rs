use super::id_selector::FaissIDSelector;
use faiss_next_sys as sys;

pub struct FaissSearchParameters {
    pub inner: *mut sys::FaissSearchParameters,
    #[allow(unused)]
    id_selector: FaissIDSelector,
}

impl Drop for FaissSearchParameters {
    fn drop(&mut self) {
        unsafe { sys::faiss_SearchParameters_free(self.inner) }
    }
}

impl FaissSearchParameters {
    pub fn new(id_selector: FaissIDSelector) -> Self {
        let mut inner = std::ptr::null_mut();
        unsafe {
            sys::faiss_SearchParameters_new(&mut inner, id_selector.inner);
        }
        Self { inner, id_selector }
    }
}
