use std::{ffi::CString, marker::PhantomData};

use faiss_next_sys::{self as ffi};

use crate::{
    error::*, impl_aux_index_structure::FaissRangeSearchResult, macros::*,
    traits::FaissIDSelectorTrait,
};

pub use ffi::FaissMetricType;
use std::ptr::null_mut;

pub trait FaissSearchParametersTrait {
    fn inner(&self) -> *mut ffi::FaissSearchParameters;
}

#[derive(Debug)]
pub struct FaissSearchParametersImpl {
    pub inner: *mut ffi::FaissSearchParameters,
}
impl_faiss_drop!(FaissSearchParametersImpl, faiss_SearchParameters_free);
impl_faiss_new!(
    FaissSearchParametersImpl,
    raw_new,
    FaissSearchParameters,
    faiss_SearchParameters_new,
    id_selector,
    *mut ffi::FaissIDSelector
);
impl FaissSearchParametersImpl {
    pub fn new(id_selector: impl FaissIDSelectorTrait) -> Result<Self> {
        let id_selector = id_selector.inner();
        Self::raw_new(id_selector)
    }
}
impl FaissSearchParametersTrait for FaissSearchParametersImpl {
    fn inner(&self) -> *mut ffi::FaissSearchParameters {
        self.inner
    }
}

pub trait FaissIndexTrait {
    fn inner(&self) -> *mut ffi::FaissIndex;

    fn clone(&self) -> Result<FaissIndexOwned> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_clone_index(self.inner(), &mut inner) })?;
        Ok(FaissIndexOwned { inner })
    }

    fn d(&self) -> i32 {
        unsafe { ffi::faiss_Index_d(self.inner()) }
    }

    fn is_trained(&self) -> bool {
        unsafe { ffi::faiss_Index_is_trained(self.inner()) > 0 }
    }
    fn ntotal(&self) -> i64 {
        unsafe { ffi::faiss_Index_ntotal(self.inner()) }
    }
    fn metric_type(&self) -> FaissMetricType {
        unsafe { ffi::faiss_Index_metric_type(self.inner()) }
    }
    fn verbose(&self) -> bool {
        unsafe { ffi::faiss_Index_verbose(self.inner()) > 0 }
    }
    fn set_verbose(&mut self, verbose: bool) {
        unsafe { ffi::faiss_Index_set_verbose(self.inner(), verbose as i32) }
    }
    fn train(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
        let n = x.as_ref().len() as i64 / self.d() as i64;
        faiss_rc(unsafe { ffi::faiss_Index_train(self.inner(), n, x.as_ref().as_ptr()) })
    }
    fn add(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
        assert_eq!(x.as_ref().len() as i32 % self.d(), 0);
        let n = x.as_ref().len() as i64 / self.d() as i64;
        faiss_rc(unsafe { ffi::faiss_Index_add(self.inner(), n, x.as_ref().as_ptr()) })
    }
    fn add_with_ids(&mut self, x: impl AsRef<[f32]>, xids: impl AsRef<[i64]>) -> Result<()> {
        assert_eq!(x.as_ref().len() as i32 % self.d(), 0);
        let n = x.as_ref().len() as i64 / self.d() as i64;
        assert_eq!(xids.as_ref().len() as i64, n);
        faiss_rc(unsafe {
            ffi::faiss_Index_add_with_ids(
                self.inner(),
                n,
                x.as_ref().as_ptr(),
                xids.as_ref().as_ptr(),
            )
        })
    }
    fn search(
        &self,
        x: impl AsRef<[f32]>,
        k: i64,
        mut distances: impl AsMut<[f32]>,
        mut labels: impl AsMut<[i64]>,
    ) -> Result<()> {
        assert_eq!(x.as_ref().len() as i32 % self.d(), 0);
        let n = x.as_ref().len() as i64 / self.d() as i64;
        assert_eq!(n * k, distances.as_mut().len() as i64);
        assert_eq!(n * k, labels.as_mut().len() as i64);
        faiss_rc(unsafe {
            ffi::faiss_Index_search(
                self.inner(),
                n,
                x.as_ref().as_ptr(),
                k,
                distances.as_mut().as_mut_ptr(),
                labels.as_mut().as_mut_ptr(),
            )
        })?;
        Ok(())
    }

    fn search_with_params(
        &self,
        x: impl AsRef<[f32]>,
        k: i64,
        params: &FaissSearchParametersImpl,
        mut distances: impl AsMut<[f32]>,
        mut labels: impl AsMut<[i64]>,
    ) -> Result<()> {
        assert_eq!(x.as_ref().len() as i32 % self.d(), 0);
        let n = x.as_ref().len() as i64 / self.d() as i64;
        assert_eq!(n * k, distances.as_mut().len() as i64);
        assert_eq!(n * k, labels.as_mut().len() as i64);
        faiss_rc(unsafe {
            ffi::faiss_Index_search_with_params(
                self.inner(),
                x.as_ref().len() as i64 / self.d() as i64,
                x.as_ref().as_ptr(),
                k,
                params.inner,
                distances.as_mut().as_mut_ptr(),
                labels.as_mut().as_mut_ptr(),
            )
        })?;
        Ok(())
    }

    fn range_search(
        &self,
        x: impl AsRef<[f32]>,
        radius: f32,
        result: &mut FaissRangeSearchResult,
    ) -> Result<()> {
        assert_eq!(x.as_ref().len() as i32 % self.d(), 0);
        let n = x.as_ref().len() as i64 / self.d() as i64;
        faiss_rc(unsafe {
            ffi::faiss_Index_range_search(
                self.inner(),
                n,
                x.as_ref().as_ptr(),
                radius,
                result.inner,
            )
        })?;
        Ok(())
    }

    fn assign(&self, x: impl AsRef<[f32]>, mut labels: impl AsMut<[i64]>, k: i64) -> Result<()> {
        assert_eq!(x.as_ref().len() as i32 % self.d(), 0);
        let n = x.as_ref().len() as i64 / self.d() as i64;
        assert_eq!(n * k, labels.as_mut().len() as i64);
        faiss_rc(unsafe {
            ffi::faiss_Index_assign(
                self.inner(),
                n,
                x.as_ref().as_ptr(),
                labels.as_mut().as_mut_ptr(),
                k,
            )
        })
    }

    fn reset(&mut self) -> Result<()> {
        faiss_rc(unsafe { ffi::faiss_Index_reset(self.inner()) })
    }

    fn remove_ids(&mut self, sel: &mut ffi::FaissIDSelector) -> Result<usize> {
        let mut nremove = 0;
        faiss_rc(unsafe { ffi::faiss_Index_remove_ids(self.inner(), sel, &mut nremove) })?;
        Ok(nremove)
    }

    fn reconstruct(&self, key: i64, mut recons: impl AsMut<[f32]>) -> Result<()> {
        assert_eq!(recons.as_mut().len() as i32 % self.d(), 0);
        faiss_rc(unsafe {
            ffi::faiss_Index_reconstruct(self.inner(), key, recons.as_mut().as_mut_ptr())
        })
    }

    fn reconstruct_n(&self, i0: i64, ni: i64, mut recons: impl AsMut<[f32]>) -> Result<()> {
        faiss_rc(unsafe {
            ffi::faiss_Index_reconstruct_n(self.inner(), i0, ni, recons.as_mut().as_mut_ptr())
        })
    }

    fn compute_residual(
        &self,
        x: impl AsRef<[f32]>,
        mut residual: impl AsMut<[f32]>,
        key: i64,
    ) -> Result<()> {
        assert_eq!(x.as_ref().len() as i32, self.d());
        assert_eq!(residual.as_mut().len() as i32, self.d());
        faiss_rc(unsafe {
            ffi::faiss_Index_compute_residual(
                self.inner(),
                x.as_ref().as_ptr(),
                residual.as_mut().as_mut_ptr(),
                key,
            )
        })
    }

    fn compute_residual_n(
        &self,
        x: impl AsRef<[f32]>,
        mut residual: impl AsMut<[f32]>,
        keys: impl AsRef<[i64]>,
    ) -> Result<()> {
        assert_eq!(x.as_ref().len(), residual.as_mut().len());
        assert_eq!(x.as_ref().len() as i32 % self.d(), 0);
        let n = x.as_ref().len() as i64 / self.d() as i64;
        assert_eq!(n, keys.as_ref().len() as i64);
        faiss_rc(unsafe {
            ffi::faiss_Index_compute_residual_n(
                self.inner(),
                n,
                x.as_ref().as_ptr(),
                residual.as_mut().as_mut_ptr(),
                keys.as_ref().as_ptr(),
            )
        })
    }

    fn sa_code_size(&self) -> Result<usize> {
        let mut ret = 0;
        faiss_rc(unsafe { ffi::faiss_Index_sa_code_size(self.inner(), &mut ret) })?;
        Ok(ret)
    }

    fn sa_decode(&self, bytes: impl AsRef<[u8]>, mut x: impl AsMut<[f32]>) -> Result<()> {
        let n = bytes.as_ref().len() as i64 / self.sa_code_size()? as i64;
        assert_eq!(x.as_mut().len() as i64, n * self.d() as i64);
        faiss_rc(unsafe {
            ffi::faiss_Index_sa_decode(
                self.inner(),
                n,
                bytes.as_ref().as_ptr(),
                x.as_mut().as_mut_ptr(),
            )
        })
    }
    fn save(&self, filename: impl AsRef<str>) -> Result<()> {
        let filename = filename.as_ref();
        let filename = CString::new(filename)?;
        faiss_rc(unsafe { ffi::faiss_write_index_fname(self.inner(), filename.as_ptr()) })
    }
}

pub struct FaissIndexBorrowed<'a, T> {
    pub inner: *mut ffi::FaissIndex,
    pub owner: PhantomData<&'a T>,
}

impl<'a, T> FaissIndexTrait for FaissIndexBorrowed<'a, T> {
    fn inner(&self) -> *mut faiss_next_sys::FaissIndex {
        self.inner
    }
}

#[derive(Debug)]
pub struct FaissIndexOwned {
    pub inner: *mut ffi::FaissIndex,
}

impl FaissIndexTrait for FaissIndexOwned {
    fn inner(&self) -> *mut ffi::FaissIndex {
        self.inner
    }
}

impl FaissIndexOwned {
    pub fn load(io_flags: i32, filename: impl AsRef<str>) -> Result<Self> {
        let filename = filename.as_ref();
        let filename = CString::new(filename)?;
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_read_index_fname(filename.as_ptr(), io_flags, &mut inner) })?;
        Ok(Self { inner })
    }
}

impl_faiss_drop!(FaissIndexOwned, faiss_Index_free);
