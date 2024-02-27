use std::ffi::{CStr, CString};
use std::ptr::null_mut;

use faiss_next_sys as sys;

use crate::{
    error::{Error, Result},
    index::IndexPtr,
    macros::faiss_rc,
};

pub struct FaissParameterRange {
    inner: *mut sys::FaissParameterRange,
}

impl FaissParameterRange {
    pub fn name(&self) -> Result<&str> {
        let name = unsafe { sys::faiss_ParameterRange_name(self.inner) };
        let name = unsafe { CStr::from_ptr(name) };
        name.to_str().map_err(|_| Error::InvalidParameterRangeName)
    }

    pub fn values(&self) -> &[f64] {
        let mut len = 0usize;
        let mut data = null_mut();
        unsafe { sys::faiss_ParameterRange_values(self.inner, &mut data, &mut len) };
        unsafe { std::slice::from_raw_parts(data, len) }
    }
}

pub struct ParameterSpace {
    inner: *mut sys::FaissParameterSpace,
}

impl Drop for ParameterSpace {
    fn drop(&mut self) {
        unsafe { sys::faiss_ParameterSpace_free(self.inner) }
    }
}

impl ParameterSpace {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_ParameterSpace_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn n_combinations(&self) -> usize {
        unsafe { sys::faiss_ParameterSpace_n_combinations(self.inner) }
    }

    pub fn combination_name(&self, idx: usize) -> Result<String> {
        let mut name = vec![0; 1000];
        faiss_rc!({
            sys::faiss_ParameterSpace_combination_name(
                self.inner,
                idx,
                name.as_mut_ptr() as *mut i8,
                1000,
            )
        })?;
        let name = CStr::from_bytes_with_nul(&name)
            .map_err(|_| Error::InvalidParameterSpaceName)
            .and_then(|v| v.to_str().map_err(|_| Error::InvalidParameterSpaceName))?
            .to_string();
        Ok(name)
    }

    pub fn set_index_parameters(
        &mut self,
        index: &mut impl IndexPtr,
        name: impl AsRef<str>,
    ) -> Result<()> {
        let name = name.as_ref();
        let name = CString::new(name).map_err(|_| Error::InvalidParameterSpaceName)?;
        faiss_rc!({
            sys::faiss_ParameterSpace_set_index_parameters(
                self.inner,
                index.mut_ptr(),
                name.as_ptr(),
            )
        })?;
        Ok(())
    }

    pub fn set_index_parameters_cno(self, index: &mut impl IndexPtr, idx: usize) -> Result<()> {
        faiss_rc!({
            sys::faiss_ParameterSpace_set_index_parameters_cno(self.inner, index.mut_ptr(), idx)
        })?;
        Ok(())
    }

    pub fn set_index_parameter(
        &mut self,
        index: &mut impl IndexPtr,
        name: impl AsRef<str>,
        value: f64,
    ) -> Result<()> {
        let name = name.as_ref();
        let name = CString::new(name).map_err(|_| Error::InvalidParameterSpaceName)?;
        faiss_rc!({
            sys::faiss_ParameterSpace_set_index_parameter(
                self.inner,
                index.mut_ptr(),
                name.as_ptr(),
                value,
            )
        })?;
        Ok(())
    }

    pub fn display(&self) {
        unsafe { sys::faiss_ParameterSpace_display(self.inner) }
    }

    pub fn add_range(&mut self, name: impl AsRef<str>) -> Result<FaissParameterRange> {
        let name = name.as_ref();
        let name = CString::new(name).map_err(|_| Error::InvalidParameterSpaceName)?;
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_ParameterSpace_add_range(self.inner, name.as_ptr(), &mut inner) })?;
        Ok(FaissParameterRange { inner })
    }
}
