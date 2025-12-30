use std::{ffi::CStr, ptr::null_mut};

use crate::{
    error::*,
    index::IndexTrait,
};
use faiss_next_sys as ffi;

/// Possible values of a parameter, sorted from least to most expensive/accurate.
#[derive(Debug)]
pub struct ParameterRange {
    pub inner: *mut ffi::FaissParameterRange,
}

impl ParameterRange {
    /// Returns the name of the parameter.
    pub fn name(&self) -> std::result::Result<&str, std::str::Utf8Error> {
        let name_ptr = unsafe { ffi::faiss_ParameterRange_name(self.inner) };
        if name_ptr.is_null() {
            // Return a default empty string or handle null case differently
            return Ok("");
        }
        let name_cstr = unsafe { CStr::from_ptr(name_ptr) };
        name_cstr.to_str()
    }

    /// Returns the values in the range.
    pub fn values(&mut self) -> &[f64] {
        let mut values_ptr = std::ptr::null_mut();
        let mut size = 0usize;
        unsafe {
            ffi::faiss_ParameterRange_values(self.inner, &mut values_ptr, &mut size);
        }
        if values_ptr.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(values_ptr, size) }
        }
    }
}

/// Uses a priori knowledge on the Faiss indexes to extract tunable parameters.
#[derive(Debug)]
pub struct ParameterSpace {
    pub inner: *mut ffi::FaissParameterSpace,
}

ffi::impl_drop!(ParameterSpace, faiss_ParameterSpace_free);

impl ParameterSpace {
    /// Creates a new ParameterSpace with default parameters.
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_ParameterSpace_new, &mut inner)?;
        Ok(Self { inner })
    }

    /// Returns the number of parameter combinations.
    pub fn n_combinations(&self) -> usize {
        unsafe { ffi::faiss_ParameterSpace_n_combinations(self.inner) }
    }

    /// Gets string representation of the combination by index.
    pub fn combination_name(&self, index: usize) -> Result<String> {
        let mut buffer = [0u8; 1000];
        ffi::ok!(
            faiss_ParameterSpace_combination_name,
            self.inner,
            index,
            buffer.as_mut_ptr() as *mut std::os::raw::c_char,
            buffer.len()
        )?;
        let cstr = unsafe { CStr::from_ptr(buffer.as_ptr() as *const std::os::raw::c_char) };
        Ok(cstr.to_string_lossy().to_string())
    }

    /// Sets a combination of parameters described by a string on an index.
    pub fn set_index_parameters<T: IndexTrait>(
        &self,
        index: &mut T,
        parameter_string: &str,
    ) -> Result<()> {
        let parameter_cstr = std::ffi::CString::new(parameter_string)?;
        ffi::ok!(
            faiss_ParameterSpace_set_index_parameters,
            self.inner,
            index.inner(),
            parameter_cstr.as_ptr()
        )?;
        Ok(())
    }

    /// Sets a combination of parameters on an index by combination index.
    pub fn set_index_parameters_cno<T: IndexTrait>(
        &self,
        index: &mut T,
        combination_index: usize,
    ) -> Result<()> {
        ffi::ok!(
            faiss_ParameterSpace_set_index_parameters_cno,
            self.inner,
            index.inner(),
            combination_index
        )?;
        Ok(())
    }

    /// Sets one of the parameters on an index.
    pub fn set_index_parameter<T: IndexTrait>(
        &self,
        index: &mut T,
        parameter_name: &str,
        value: f64,
    ) -> Result<()> {
        let parameter_name_cstr = std::ffi::CString::new(parameter_name)?;
        ffi::ok!(
            faiss_ParameterSpace_set_index_parameter,
            self.inner,
            index.inner(),
            parameter_name_cstr.as_ptr(),
            value
        )?;
        Ok(())
    }

    /// Prints a description of the parameter space to stdout.
    pub fn display(&self) {
        unsafe {
            ffi::faiss_ParameterSpace_display(self.inner);
        }
    }

    /// Adds a new parameter range (or returns it if it exists).
    pub fn add_range(&mut self, name: &str) -> Result<ParameterRange> {
        let name_cstr = std::ffi::CString::new(name)?;
        let mut range_inner = null_mut();
        ffi::ok!(
            faiss_ParameterSpace_add_range,
            self.inner,
            name_cstr.as_ptr(),
            &mut range_inner
        )?;
        Ok(ParameterRange { inner: range_inner })
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
// 
//     #[test]
//     fn test_parameter_space() -> Result<()> {
//         let space = ParameterSpace::new()?;
//         assert_eq!(space.n_combinations(), 0);
//         
//         // Test getting a combination name (should fail gracefully if no combinations available)
//         let result = space.combination_name(0);
//         // This might fail with an out-of-bounds error, which is expected if no combinations exist
//         if result.is_ok() {
//             println!("First combination: {}", result.unwrap());
//         }
//         
//         // Display parameter space
//         space.display();
//         
//         Ok(())
//     }
// }
