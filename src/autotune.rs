use faiss_next_sys as sys;

pub struct ParameterRange {
    inner: sys::FaissParameterRange,
}

impl ParameterRange {
    pub fn name(&self) -> &str {
        unsafe {
            let c_str = std::ffi::CStr::from_ptr(self.inner.name);
            c_str.to_str().unwrap()
        }
    }
}
