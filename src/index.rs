// use std::{ffi::CString, marker::PhantomData};
// use crate::{
//     error::*, impl_aux_index_structure::FaissRangeSearchResult, macros::*,
//     traits::FaissIDSelectorTrait,
// };

use crate::error::Result;
use crate::impl_aux_index_structures::{IDSelectorTrait, RangeSearchResult};
use faiss_next_sys as ffi;
use std::marker::PhantomData;
use std::ptr::null_mut;

pub trait SearchParametersTrait: std::fmt::Debug {
    fn inner(&self) -> *mut ffi::FaissSearchParameters;
}

#[derive(Debug)]
pub struct SearchParameters {
    inner: *mut ffi::FaissSearchParameters,
}

ffi::impl_drop!(SearchParameters, faiss_SearchParameters_free);

impl SearchParameters {
    pub fn new(sel: impl IDSelectorTrait) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_SearchParameters_new, &mut inner, sel.inner())?;
        Ok(Self { inner })
    }
}

impl SearchParametersTrait for SearchParameters {
    fn inner(&self) -> *mut ffi::FaissSearchParameters {
        self.inner
    }
}

pub use ffi::FaissMetricType as MetricType;

pub trait IndexTrait: std::fmt::Debug {
    fn inner(&self) -> *mut ffi::FaissIndex;

    fn d(&self) -> i32 {
        ffi::run!(faiss_Index_d, self.inner())
    }

    fn is_trained(&self) -> bool {
        ffi::run!(faiss_Index_is_trained, self.inner()) > 0
    }

    fn ntotal(&self) -> i64 {
        ffi::run!(faiss_Index_ntotal, self.inner())
    }

    fn metric_type(&self) -> MetricType {
        ffi::run!(faiss_Index_metric_type, self.inner())
    }

    fn verbose(&self) -> bool {
        ffi::run!(faiss_Index_verbose, self.inner()) > 0
    }

    fn set_verbose(&mut self, verbose: bool) {
        ffi::run!(faiss_Index_set_verbose, self.inner(), verbose as i32)
    }

    fn train(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
        let x = x.as_ref();
        assert_eq!(
            x.len() % self.d() as usize,
            0,
            "x.len() must be a multiple of index.d()"
        );
        let n = x.as_ref().len() as i64 / self.d() as i64;
        ffi::ok!(faiss_Index_train, self.inner(), n, x.as_ref().as_ptr())?;
        Ok(())
    }

    fn add(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
        let x = x.as_ref();
        assert_eq!(
            x.len() % self.d() as usize,
            0,
            "x.len() must be a multiple of index.d()"
        );
        let n = x.as_ref().len() as i64 / self.d() as i64;
        ffi::ok!(faiss_Index_add, self.inner(), n, x.as_ref().as_ptr())?;
        Ok(())
    }

    fn add_with_ids(&mut self, x: impl AsRef<[f32]>, ids: impl AsRef<[i64]>) -> Result<()> {
        let x = x.as_ref();
        let ids = ids.as_ref();

        assert_eq!(
            x.len() % self.d() as usize,
            0,
            "x.len() must be a multiple of index.d()"
        );

        let n = x.as_ref().len() as i64 / self.d() as i64;

        assert_eq!(
            ids.len() as i64,
            n,
            "ids.len() must be equal to x.len() / index.d()"
        );

        ffi::ok!(
            faiss_Index_add_with_ids,
            self.inner(),
            n,
            x.as_ref().as_ptr(),
            ids.as_ref().as_ptr()
        )?;
        Ok(())
    }

    fn search(
        &self,
        x: impl AsRef<[f32]>,
        k: i64,
        mut distances: impl AsMut<[f32]>,
        mut labels: impl AsMut<[i64]>,
    ) -> Result<()> {
        let x = x.as_ref();
        assert_eq!(
            x.len() % self.d() as usize,
            0,
            "x.len() must be a multiple of index.d()"
        );

        let n = x.as_ref().len() as i64 / self.d() as i64;

        assert_eq!(
            distances.as_mut().len() as i64,
            n * k,
            "distances.len() must be equal to n * k"
        );
        assert_eq!(
            labels.as_mut().len() as i64,
            n * k,
            "labels.len() must be equal to n * k"
        );

        ffi::ok!(
            faiss_Index_search,
            self.inner(),
            n,
            x.as_ref().as_ptr(),
            k,
            distances.as_mut().as_mut_ptr(),
            labels.as_mut().as_mut_ptr()
        )?;

        Ok(())
    }

    fn search_with_params(
        &self,
        x: impl AsRef<[f32]>,
        k: i64,
        params: &impl SearchParametersTrait,
        mut distances: impl AsMut<[f32]>,
        mut labels: impl AsMut<[i64]>,
    ) -> Result<()> {
        let x = x.as_ref();
        assert_eq!(
            x.len() % self.d() as usize,
            0,
            "x.len() must be a multiple of index.d()"
        );
        let n = x.len() as i64 / self.d() as i64;

        assert_eq!(distances.as_mut().len() as i64, n * k);
        assert_eq!(labels.as_mut().len() as i64, n * k);

        ffi::ok!(
            faiss_Index_search_with_params,
            self.inner(),
            n,
            x.as_ref().as_ptr(),
            k,
            params.inner(),
            distances.as_mut().as_mut_ptr(),
            labels.as_mut().as_mut_ptr()
        )?;
        Ok(())
    }

    fn range_search(
        &self,
        x: impl AsRef<[f32]>,
        radius: f32,
        result: &mut RangeSearchResult,
    ) -> Result<()> {
        let x = x.as_ref();
        assert_eq!(
            x.len() % self.d() as usize,
            0,
            "x.len() must be a multiple of index.d()"
        );
        let n = x.as_ref().len() as i64 / self.d() as i64;
        ffi::ok!(
            faiss_Index_range_search,
            self.inner(),
            n,
            x.as_ref().as_ptr(),
            radius,
            result.inner
        )?;
        Ok(())
    }

    fn assign(&self, x: impl AsRef<[f32]>, mut labels: impl AsMut<[i64]>, k: i64) -> Result<()> {
        let x = x.as_ref();
        assert_eq!(
            x.len() % self.d() as usize,
            0,
            "x.len() must be a multiple of index.d()"
        );
        let n = x.as_ref().len() as i64 / self.d() as i64;
        assert_eq!(n * k, labels.as_mut().len() as i64);
        ffi::ok!(
            faiss_Index_assign,
            self.inner(),
            n,
            x.as_ref().as_ptr(),
            labels.as_mut().as_mut_ptr(),
            k
        )?;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        ffi::ok!(faiss_Index_reset, self.inner())?;
        Ok(())
    }

    fn remove_ids(&mut self, sel: &mut impl IDSelectorTrait) -> Result<usize> {
        let mut nremoved = 0usize;
        ffi::ok!(
            faiss_Index_remove_ids,
            self.inner(),
            sel.inner(),
            &mut nremoved
        )?;
        Ok(nremoved)
    }

    fn reconstruct(&self, key: i64, mut recons: impl AsMut<[f32]>) -> Result<()> {
        let recons = recons.as_mut();
        assert_eq!(
            recons.len() % self.d() as usize,
            0,
            "x.len() must be a multiple of index.d()"
        );
        ffi::ok!(
            faiss_Index_reconstruct,
            self.inner(),
            key,
            recons.as_mut_ptr()
        )?;
        Ok(())
    }

    fn reconstruct_n(&self, i0: i64, mut recons: impl AsMut<[f32]>) -> Result<()> {
        let recons = recons.as_mut();
        assert_eq!(
            recons.len() % self.d() as usize,
            0,
            "x.len() must be a multiple of index.d()"
        );
        let ni = recons.len() as i64 / self.d() as i64;
        ffi::ok!(
            faiss_Index_reconstruct_n,
            self.inner(),
            i0,
            ni,
            recons.as_mut().as_mut_ptr()
        )?;
        Ok(())
    }

    fn compute_residual(
        &self,
        x: impl AsRef<[f32]>,
        mut residual: impl AsMut<[f32]>,
        key: i64,
    ) -> Result<()> {
        let x = x.as_ref();
        assert_eq!(
            x.len() % self.d() as usize,
            0,
            "x.len() must be a multiple of index.d()"
        );

        ffi::ok!(
            faiss_Index_compute_residual,
            self.inner(),
            x.as_ref().as_ptr(),
            residual.as_mut().as_mut_ptr(),
            key
        )?;

        Ok(())
    }

    fn compute_residual_n(
        &self,
        x: impl AsRef<[f32]>,
        mut residual: impl AsMut<[f32]>,
        keys: impl AsRef<[i64]>,
    ) -> Result<()> {
        let x = x.as_ref();
        assert_eq!(
            x.len() % self.d() as usize,
            0,
            "x.len() must be a multiple of index.d()"
        );
        let n = x.len() as i64 / self.d() as i64;
        ffi::ok!(
            faiss_Index_compute_residual_n,
            self.inner(),
            n,
            x.as_ptr(),
            residual.as_mut().as_mut_ptr(),
            keys.as_ref().as_ptr()
        )?;
        Ok(())
    }

    fn sa_code_size(&self) -> Result<usize> {
        let mut ret = 0;
        ffi::ok!(faiss_Index_sa_code_size, self.inner(), &mut ret)?;
        Ok(ret)
    }

    fn sa_decode(&self, bytes: impl AsRef<[u8]>, mut x: impl AsMut<[f32]>) -> Result<()> {
        let n = bytes.as_ref().len() as i64 / self.sa_code_size()? as i64;
        assert_eq!(x.as_mut().len() as i64, n * self.d() as i64);
        ffi::ok!(
            faiss_Index_sa_decode,
            self.inner(),
            n,
            bytes.as_ref().as_ptr(),
            x.as_mut().as_mut_ptr()
        )?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct IndexBorrowed<'a> {
    inner: *mut ffi::FaissIndex,
    _marker: PhantomData<&'a ()>,
}

impl<'a> IndexBorrowed<'a> {
    pub fn new(inner: *mut ffi::FaissIndex) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

// impl<T> std::fmt::Debug for IndexBorrowed<'_, T> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("IndexBorrowed")
//             .field("inner", &self.inner)
//             .finish()
//     }
// }

// impl<'a, T> Drop for IndexBorrowed<'a, T> {
//     fn drop(&mut self) {
//         ffi::run!(faiss_Index_free, self.inner());
//     }
// }

impl<'a> IndexTrait for IndexBorrowed<'a> {
    fn inner(&self) -> *mut faiss_next_sys::FaissIndex {
        self.inner
    }
}

#[derive(Debug)]
pub struct IndexOwned {
    inner: *mut ffi::FaissIndex,
}

impl IndexOwned {
    pub(crate) fn new(inner: *mut ffi::FaissIndex) -> Self {
        Self { inner }
    }
}

ffi::impl_drop!(IndexOwned, faiss_Index_free);

impl IndexTrait for IndexOwned {
    fn inner(&self) -> *mut faiss_next_sys::FaissIndex {
        self.inner
    }
}

// pub trait FaissIndexTrait {
//     #[cfg(all(feature = "cuda", not(target_os = "macos")))]
//     fn to_gpu(
//         &self,
//         provider: impl FaissGpuResourcesProviderTrait,
//         device: i32,
//     ) -> Result<FaissGpuIndexOwned> {
//         let mut inner = null_mut();
//         faiss_rc(unsafe {
//             ffi::faiss_index_cpu_to_gpu(provider.inner(), device, self.inner(), &mut inner)
//         })?;
//         Ok(FaissGpuIndexOwned { inner })
//     }

//     #[cfg(all(feature = "cuda", not(target_os = "macos")))]
//     fn to_gpu_with_options(
//         &self,
//         provider: impl FaissGpuResourcesProviderTrait,
//         options: impl FaissGpuClonerOptionsTrait,
//         device: i32,
//     ) -> Result<FaissGpuIndexOwned> {
//         let mut inner = null_mut();
//         faiss_rc(unsafe {
//             ffi::faiss_index_cpu_to_gpu_with_options(
//                 provider.inner(),
//                 device,
//                 self.inner(),
//                 options.inner(),
//                 &mut inner,
//             )
//         })?;
//         Ok(FaissGpuIndexOwned { inner })
//     }

//     #[cfg(all(feature = "cuda", not(target_os = "macos")))]
//     fn to_gpu_multiple<P: FaissGpuResourcesProviderTrait>(
//         &self,
//         providers: impl IntoIterator<Item = P>,
//         devices: impl AsRef<[i32]>,
//     ) -> Result<FaissGpuIndexOwned> {
//         let mut inner = null_mut();
//         let providers = providers.into_iter().map(|p| p.inner()).collect::<Vec<_>>();
//         faiss_rc(unsafe {
//             ffi::faiss_index_cpu_to_gpu_multiple(
//                 providers.as_ptr(),
//                 devices.as_ref().as_ptr(),
//                 devices.as_ref().len(),
//                 self.inner(),
//                 &mut inner,
//             )
//         })?;
//         Ok(FaissGpuIndexOwned { inner })
//     }

//     #[cfg(all(feature = "cuda", not(target_os = "macos")))]
//     fn to_gpu_multiple_with_options<P: FaissGpuResourcesProviderTrait>(
//         &self,
//         providers: impl IntoIterator<Item = P>,
//         devices: impl AsRef<[i32]>,
//         options: FaissGpuMultipleClonerOptions,
//     ) -> Result<FaissGpuIndexOwned> {
//         let mut inner = null_mut();
//         let providers = providers.into_iter().map(|p| p.inner()).collect::<Vec<_>>();
//         faiss_rc(unsafe {
//             ffi::faiss_index_cpu_to_gpu_multiple_with_options(
//                 providers.as_ptr(),
//                 providers.len(),
//                 devices.as_ref().as_ptr(),
//                 devices.as_ref().len(),
//                 self.inner(),
//                 options.inner,
//                 &mut inner,
//             )
//         })?;
//         Ok(FaissGpuIndexOwned { inner })
//     }

//     fn clone(&self) -> Result<FaissIndexOwned> {
//         let mut inner = null_mut();
//         faiss_rc(unsafe { ffi::faiss_clone_index(self.inner(), &mut inner) })?;
//         Ok(FaissIndexOwned { inner })
//     }
//     fn save(&self, filename: impl AsRef<str>) -> Result<()> {
//         let filename = filename.as_ref();
//         let filename = CString::new(filename)?;
//         faiss_rc(unsafe { ffi::faiss_write_index_fname(self.inner(), filename.as_ptr()) })
//     }
// }

// pub struct FaissIndexBorrowed<'a, T> {
//     pub inner: *mut ffi::FaissIndex,
//     pub owner: PhantomData<&'a T>,
// }

// impl<'a, T> FaissIndexTrait for FaissIndexBorrowed<'a, T> {
//     fn inner(&self) -> *mut faiss_next_sys::FaissIndex {
//         self.inner
//     }
// }

// #[derive(Debug)]
// pub struct FaissIndexOwned {
//     pub inner: *mut ffi::FaissIndex,
// }

// impl FaissIndexTrait for FaissIndexOwned {
//     fn inner(&self) -> *mut ffi::FaissIndex {
//         self.inner
//     }
// }

// impl FaissIndexOwned {
//     pub fn load(io_flags: i32, filename: impl AsRef<str>) -> Result<Self> {
//         let filename = filename.as_ref();
//         let filename = CString::new(filename)?;
//         let mut inner = null_mut();
//         faiss_rc(unsafe { ffi::faiss_read_index_fname(filename.as_ptr(), io_flags, &mut inner) })?;
//         Ok(Self { inner })
//     }
// }

// impl_faiss_drop!(FaissIndexOwned, faiss_Index_free);
