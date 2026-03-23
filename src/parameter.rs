use std::ffi::CString;
use std::ptr;

use faiss_next_sys::{self, FaissParameterSpace};

use crate::error::{check_return_code, Result};
use crate::index::Index;

pub struct ParameterSpace {
    ptr: *mut FaissParameterSpace,
}

impl ParameterSpace {
    pub fn new() -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissParameterSpace = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_ParameterSpace_new(&mut ptr))?;
            Ok(Self { ptr })
        }
    }

    pub fn n_combinations(&self) -> usize {
        unsafe { faiss_next_sys::faiss_ParameterSpace_n_combinations(self.ptr) }
    }

    pub fn combination_name(&self, cno: usize) -> Result<String> {
        let mut name_buf = vec![0i8; 256];
        check_return_code(unsafe {
            faiss_next_sys::faiss_ParameterSpace_combination_name(
                self.ptr,
                cno,
                name_buf.as_mut_ptr(),
                name_buf.len(),
            )
        })?;
        let name = unsafe {
            std::ffi::CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        Ok(name)
    }

    pub fn set_index_parameters<I: Index>(
        &self,
        index: &mut I,
        parameter_string: &str,
    ) -> Result<()> {
        let c_param = CString::new(parameter_string)?;
        check_return_code(unsafe {
            faiss_next_sys::faiss_ParameterSpace_set_index_parameters(
                self.ptr,
                index.inner_ptr(),
                c_param.as_ptr(),
            )
        })
    }

    pub fn set_index_parameters_cno<I: Index>(&self, index: &mut I, cno: usize) -> Result<()> {
        check_return_code(unsafe {
            faiss_next_sys::faiss_ParameterSpace_set_index_parameters_cno(
                self.ptr,
                index.inner_ptr(),
                cno,
            )
        })
    }

    pub fn set_index_parameter<I: Index>(
        &self,
        index: &mut I,
        name: &str,
        value: f64,
    ) -> Result<()> {
        let c_name = CString::new(name)?;
        check_return_code(unsafe {
            faiss_next_sys::faiss_ParameterSpace_set_index_parameter(
                self.ptr,
                index.inner_ptr(),
                c_name.as_ptr(),
                value,
            )
        })
    }

    pub fn display(&self) {
        unsafe { faiss_next_sys::faiss_ParameterSpace_display(self.ptr) }
    }

    pub fn add_range(&mut self, name: &str) -> Result<()> {
        let c_name = CString::new(name)?;
        unsafe {
            let mut range_ptr: *mut faiss_next_sys::FaissParameterRange = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_ParameterSpace_add_range(
                self.ptr,
                c_name.as_ptr(),
                &mut range_ptr,
            ))
        }
    }
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self::new().expect("Failed to create ParameterSpace")
    }
}

impl Drop for ParameterSpace {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_ParameterSpace_free(self.ptr);
            }
        }
    }
}
