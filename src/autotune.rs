use faiss_next_sys as sys;
pub struct ParameterRange {
    inner: *mut sys::FaissParameterRange,
}

impl Drop for ParameterRange {
    fn drop(&mut self) {
        unsafe {
            sys::faiss_ParameterRange_free(self.inner);
        }
    }
}
