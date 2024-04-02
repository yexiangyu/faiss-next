use std::{
    ffi::{CStr, CString},
    ptr::null_mut,
    slice::from_raw_parts,
};

use crate::error::Result;
use crate::index::IndexTrait;
use crate::macros::rc;
use tracing::trace;

use faiss_next_sys as sys;

pub struct ParameterRange {
    inner: *mut sys::FaissParameterRange,
}

impl ParameterRange {
    pub fn name(&self) -> &str {
        unsafe {
            let c_str = sys::faiss_ParameterRange_name(self.inner);
            let c_str = CStr::from_ptr(c_str);
            c_str.to_str().expect("?")
        }
    }

    pub fn values(&self) -> &[f64] {
        unsafe {
            let mut data = null_mut();
            let mut len = 0usize;
            sys::faiss_ParameterRange_values(self.inner, &mut data, &mut len);
            from_raw_parts(data, len)
        }
    }
}

impl std::fmt::Debug for ParameterRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParameterRange")
            .field("inner", &self.inner)
            .field("name", &self.name())
            .finish()
    }
}

pub struct ParameterSpace {
    inner: *mut sys::FaissParameterSpace,
}

impl std::fmt::Debug for ParameterSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParameterSpace")
            .field("inner", &self.inner)
            .field("n_combinations", &self.n_combinations())
            .finish()
    }
}

impl Drop for ParameterSpace {
    fn drop(&mut self) {
        unsafe {
            trace!(?self, "drop");
            sys::faiss_ParameterSpace_free(self.inner);
        }
    }
}

impl ParameterSpace {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_ParameterSpace_new(&mut inner) })?;
        let r = Self { inner };
        trace!(?r, "new");
        Ok(r)
    }

    pub fn n_combinations(&self) -> usize {
        unsafe { sys::faiss_ParameterSpace_n_combinations(self.inner) }
    }

    pub fn combination_name(&self, n: usize) -> Result<String> {
        // TODO: comfirm parameter type
        unsafe {
            let mut c_str_block = vec![0i8; 4096];
            sys::faiss_ParameterSpace_combination_name(
                self.inner,
                4096,
                c_str_block.as_mut_ptr(),
                n,
            );
            let c_str = CString::from_raw(c_str_block.as_mut_ptr());
            Ok(c_str.to_string_lossy().into())
        }
    }

    pub fn set_index_parameters(
        &self,
        index: &mut impl IndexTrait,
        arg3: impl AsRef<str>,
    ) -> Result<()> {
        let c_str = CString::new(arg3.as_ref()).expect("?");
        rc!({
            sys::faiss_ParameterSpace_set_index_parameters(self.inner, index.ptr(), c_str.as_ptr())
        })
    }

    pub fn set_index_parameter(
        &self,
        index: &mut impl IndexTrait,
        arg3: impl AsRef<str>,
        arg4: f64,
    ) -> Result<()> {
        let c_str = CString::new(arg3.as_ref()).expect("?");
        rc!({
            sys::faiss_ParameterSpace_set_index_parameter(
                self.inner,
                index.ptr(),
                c_str.as_ptr(),
                arg4,
            )
        })
    }

    pub fn set_index_parameters_cno(&self, index: &mut impl IndexTrait, arg3: usize) -> Result<()> {
        rc!({ sys::faiss_ParameterSpace_set_index_parameters_cno(self.inner, index.ptr(), arg3) })
    }

    pub fn display(&self) {
        unsafe {
            sys::faiss_ParameterSpace_display(self.inner);
        }
    }

    pub fn add_range(&mut self, name: impl AsRef<str>) -> Result<ParameterRange> {
        let c_str = CString::new(name.as_ref()).expect("?");
        let mut inner = null_mut();
        rc!({ sys::faiss_ParameterSpace_add_range(self.inner, c_str.as_ptr(), &mut inner) })?;
        Ok(ParameterRange { inner })
    }
}
