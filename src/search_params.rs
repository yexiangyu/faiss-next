use std::ptr;

use faiss_next_sys::{self, FaissSearchParameters, FaissSearchParametersIVF};

use crate::error::{check_return_code, Result};

pub trait SearchParams {
    fn as_ptr(&self) -> *const FaissSearchParameters;
}

pub struct SearchParameters {
    ptr: *mut FaissSearchParameters,
}

impl SearchParameters {
    pub fn new() -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissSearchParameters = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_SearchParameters_new(
                &mut ptr,
                ptr::null_mut(),
            ))?;
            Ok(Self { ptr })
        }
    }
}

impl SearchParams for SearchParameters {
    fn as_ptr(&self) -> *const FaissSearchParameters {
        self.ptr
    }
}

impl Default for SearchParameters {
    fn default() -> Self {
        Self::new().expect("Failed to create SearchParameters")
    }
}

impl Drop for SearchParameters {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_SearchParameters_free(self.ptr);
            }
        }
    }
}

pub struct SearchParametersIvf {
    ptr: *mut FaissSearchParametersIVF,
}

impl SearchParametersIvf {
    pub fn new() -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissSearchParametersIVF = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_SearchParametersIVF_new(&mut ptr))?;
            Ok(Self { ptr })
        }
    }

    pub fn with_params(nprobe: usize, max_codes: usize) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissSearchParametersIVF = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_SearchParametersIVF_new_with(
                &mut ptr,
                ptr::null_mut(),
                nprobe,
                max_codes,
            ))?;
            Ok(Self { ptr })
        }
    }

    pub fn nprobe(&self) -> usize {
        unsafe { faiss_next_sys::faiss_SearchParametersIVF_nprobe(self.ptr) }
    }

    pub fn set_nprobe(&mut self, nprobe: usize) {
        unsafe { faiss_next_sys::faiss_SearchParametersIVF_set_nprobe(self.ptr, nprobe) }
    }

    pub fn max_codes(&self) -> usize {
        unsafe { faiss_next_sys::faiss_SearchParametersIVF_max_codes(self.ptr) }
    }

    pub fn set_max_codes(&mut self, max_codes: usize) {
        unsafe { faiss_next_sys::faiss_SearchParametersIVF_set_max_codes(self.ptr, max_codes) }
    }
}

impl SearchParams for SearchParametersIvf {
    fn as_ptr(&self) -> *const FaissSearchParameters {
        self.ptr as *const FaissSearchParameters
    }
}

impl Default for SearchParametersIvf {
    fn default() -> Self {
        Self::new().expect("Failed to create SearchParametersIvf")
    }
}

impl Drop for SearchParametersIvf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_SearchParametersIVF_free(self.ptr);
            }
        }
    }
}
