use crate::error::{Error, Result};
use crate::index::common::FaissIndexTrait;
use crate::rc;
use faiss_next_sys as sys;
use std::ffi::CString;
use std::{
    ffi::CStr,
    ptr::{addr_of_mut, null_mut},
};

pub struct FaissParameterRange {
    inner: *mut sys::FaissParameterRange,
}
impl FaissParameterRange {
    pub fn values(&self) -> &[f64] {
        let mut data = null_mut();
        let mut size = 0usize;
        unsafe {
            sys::faiss_ParameterRange_values(self.inner, addr_of_mut!(data), addr_of_mut!(size))
        };
        unsafe { std::slice::from_raw_parts(data, size) }
    }

    pub fn name(&self) -> Result<String> {
        let name = unsafe { sys::faiss_ParameterRange_name(self.inner) };
        let name = unsafe { CStr::from_ptr(name) };
        Ok(name
            .to_str()
            .map_err(|_| Error::InvalidIndexParameters)?
            .to_string())
    }
}

pub struct FaissParameterSpace {
    inner: *mut sys::FaissParameterSpace,
}

impl FaissParameterSpace {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_ParameterSpace_new(addr_of_mut!(inner)) })?;
        Ok(FaissParameterSpace { inner })
    }

    pub fn n_combinations(&self) -> usize {
        unsafe { sys::faiss_ParameterSpace_n_combinations(self.inner) }
    }

    pub fn combination_name(&self, n: usize) -> Result<String> {
        let mut buffer = [0i8; 1000];
        rc!({
            sys::faiss_ParameterSpace_combination_name(self.inner, n, buffer.as_mut_ptr(), 1000)
        })?;
        let name = unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_str()
            .map_err(|_| Error::InvalidCombinationName)?
            .to_string();
        Ok(name)
    }

    pub fn set_index_parameters(
        &self,
        index: &mut impl FaissIndexTrait,
        params: &str,
    ) -> Result<()> {
        let params = CString::new(params).map_err(|_| Error::InvalidIndexParameters)?;
        rc!({
            sys::faiss_ParameterSpace_set_index_parameters(
                self.inner,
                index.inner(),
                params.as_ptr(),
            )
        })?;
        Ok(())
    }

    pub fn set_index_parameters_by_cno(
        &self,
        index: &mut impl FaissIndexTrait,
        n: usize,
    ) -> Result<()> {
        rc!({ sys::faiss_ParameterSpace_set_index_parameters_cno(self.inner, index.inner(), n) })?;
        Ok(())
    }

    pub fn set_index_parameter(
        &self,
        index: &mut impl FaissIndexTrait,
        name: &str,
        value: f64,
    ) -> Result<()> {
        let name = CString::new(name).map_err(|_| Error::InvalidIndexParameters)?;
        rc!({
            sys::faiss_ParameterSpace_set_index_parameter(
                self.inner,
                index.inner(),
                name.as_ptr(),
                value,
            )
        })?;
        Ok(())
    }
}
