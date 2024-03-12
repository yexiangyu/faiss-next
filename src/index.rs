#[cxx::bridge]
#[allow(clippy::missing_safety_doc)]
pub mod ffi {
    unsafe extern "C++" {
        include!("faiss-next/src/cpp/index.hpp");
        fn version_major() -> i32;
        fn version_minor() -> i32;
        fn version_patch() -> i32;
        unsafe fn index_free(index: *mut i32);
        unsafe fn index_d(index: *const i32) -> i32;
        unsafe fn index_ntotal(index: *const i32) -> i64;
        unsafe fn index_verbose(index: *const i32) -> bool;
        unsafe fn index_set_verbose(index: *mut i32, verbose: bool);
        unsafe fn index_is_trained(index: *const i32) -> bool;
        unsafe fn index_metric_type(index: *const i32) -> i32;
        unsafe fn index_metric_arg(index: *const i32) -> f32;
        unsafe fn index_train(index: *mut i32, n: i64, x: *const f32);
        unsafe fn index_add(index: *mut i32, n: i64, x: *const f32);
        unsafe fn index_add_with_ids(index: *mut i32, n: i64, x: *const f32, xids: *const i64);
        unsafe fn index_search(
            index: *const i32,
            n: i64,
            x: *const f32,
            k: i64,
            distances: *mut f32,
            labels: *mut i64,
            params: *const i32,
        );
        unsafe fn index_range_search(
            index: *const i32,
            n: i64,
            x: *const f32,
            radius: f32,
            result: *mut i32,
            params: *const i32,
        );
        unsafe fn index_assign(index: *const i32, n: i64, x: *const f32, labels: *mut i64, k: i64);
        unsafe fn index_reset(index: *mut i32);
        unsafe fn index_remove_ids(index: *mut i32, sel: *const i32) -> usize;
        unsafe fn index_reconstruct(index: *mut i32, key: i64, recons: *mut f32);
        unsafe fn index_reconstruct_batch(
            index: *mut i32,
            n: i64,
            keys: *const i64,
            recons: *mut f32,
        );
        unsafe fn index_reconstruct_n(index: *mut i32, i0: i64, ni: i64, recons: *mut f32);
        #[allow(clippy::too_many_arguments)]
        unsafe fn index_search_and_reconstruct(
            index: *mut i32,
            n: i64,
            x: *const f32,
            k: i64,
            distances: *mut f32,
            labels: *mut i64,
            recons: *mut f32,
            params: *const i32,
        );
        unsafe fn index_compute_residual(
            index: *const i32,
            x: *const f32,
            residuals: *mut f32,
            key: i64,
        );
        unsafe fn index_compute_residual_n(
            index: *const i32,
            n: i64,
            x: *const f32,
            residuals: *mut f32,
            keys: *const i64,
        );
        unsafe fn index_get_distance_computer(index: *const i32) -> *mut i32;
        unsafe fn index_sa_code_size(index: *const i32) -> usize;
        unsafe fn index_sa_encode(index: *const i32, n: i64, x: *const f32, bytes: *mut u8);
        unsafe fn index_sa_decode(index: *const i32, n: i64, bytes: *const u8, x: *mut f32);
        unsafe fn index_merge_from(index: *mut i32, rhs: *mut i32, add_id: i64);
        unsafe fn index_check_compatible_for_merge(index: *const i32, other: *const i32) -> bool;
        unsafe fn search_parameters_new(sel: *mut i32) -> *mut i32;
        unsafe fn search_parameters_free(sp: *mut i32);
        unsafe fn search_parameters_sel(sp: *mut i32) -> *mut i32;
        unsafe fn search_parameters_set_sel(sp: *mut i32, sel: *mut i32);
    }
}

pub fn version() -> (i32, i32, i32) {
    (
        ffi::version_major(),
        ffi::version_minor(),
        ffi::version_patch(),
    )
}

macro_rules! impl_index_drop {
    ($cls: ty) => {
        impl Drop for $cls {
            fn drop(&mut self) {
                if self.inner.is_null() {
                    tracing::trace!("drop index inner={:?}", self.inner);
                    unsafe { crate::index::ffi::index_free(self.inner) };
                }
            }
        }
    };

    ($cls: ty, $free: path) => {
        impl Drop for $cls {
            fn drop(&mut self) {
                if self.inner.is_null() {
                    tracing::trace!("drop index inner={:?}", self.inner);
                    unsafe { $free(self.inner) };
                }
            }
        }
    };
}

pub(crate) use impl_index_drop;

macro_rules! impl_index_trait {
    ($cls: ty) => {
        impl crate::index::IndexTrait for $cls {
            fn ptr(&self) -> crate::index::IndexPtr {
                self.inner
            }

            fn into_ptr(self) -> crate::index::IndexPtr {
                let mut s = self;
                let inner = s.inner;
                s.inner = std::ptr::null_mut();
                inner
            }
        }
    };
}

pub(crate) use impl_index_trait;

use std::{marker::PhantomData, ptr::null_mut};

use crate::{
    aux_index_structures::RangeSearchResult,
    distance_computer::{DistanceComputerPtr, DistanceComputerTrait},
    error::{Error, Result},
    id_selector::IDSelectorTrait,
    metric::MetricType,
};

pub type IndexPtr = *mut i32;

pub struct IndexDistanceComputer<'a, T>
where
    T: IndexTrait,
{
    inner: DistanceComputerPtr,
    marker: PhantomData<&'a T>,
}

impl<T> DistanceComputerTrait for IndexDistanceComputer<'_, T>
where
    T: IndexTrait,
{
    fn ptr(&self) -> DistanceComputerPtr {
        self.inner
    }
}

pub trait IndexTrait {
    fn ptr(&self) -> IndexPtr;
    fn into_ptr(self) -> IndexPtr;

    fn d(&self) -> i32 {
        unsafe { ffi::index_d(self.ptr()) }
    }

    fn ntotal(&self) -> i64 {
        unsafe { ffi::index_ntotal(self.ptr()) }
    }

    fn verbose(&self) -> bool {
        unsafe { ffi::index_verbose(self.ptr()) }
    }

    fn set_verbose(&mut self, verbose: bool) {
        unsafe { ffi::index_set_verbose(self.ptr(), verbose) }
    }

    fn is_trained(&self) -> bool {
        unsafe { ffi::index_is_trained(self.ptr()) }
    }

    fn metric_type(&self) -> Result<MetricType> {
        let m = unsafe { ffi::index_metric_type(self.ptr()) };
        MetricType::try_from(m).map_err(|_| Error::MetricFromNumber(m))
    }

    fn metric_arg(&self) -> f32 {
        unsafe { ffi::index_metric_arg(self.ptr()) }
    }

    fn train(&mut self, x: impl AsRef<[f32]>) {
        let x = x.as_ref();
        assert_eq!(x.len() as i32 % self.d(), 0);
        let n = self.d() as i64 / x.len() as i64;
        unsafe { ffi::index_train(self.ptr(), n, x.as_ptr()) }
    }

    fn add(&mut self, x: impl AsRef<[f32]>) {
        let x = x.as_ref();
        assert_eq!(x.len() as i32 % self.d(), 0);
        let n = self.d() as i64 / x.len() as i64;
        unsafe { ffi::index_add(self.ptr(), n, x.as_ptr()) }
    }

    fn add_with_ids(&mut self, x: impl AsRef<[f32]>, xids: impl AsRef<[i64]>) {
        let x = x.as_ref();
        let xids = xids.as_ref();
        assert_eq!(x.len() as i32 % self.d(), 0);
        assert_eq!(x.len(), xids.len());
        let n = self.d() as i64 / x.len() as i64;
        unsafe { ffi::index_add_with_ids(self.ptr(), n, x.as_ptr(), xids.as_ptr()) }
    }

    fn search(
        &self,
        x: impl AsRef<[f32]>,
        k: i64,
        mut distances: impl AsMut<[f32]>,
        mut labels: impl AsMut<[i64]>,
        params: Option<&impl SearchParametersTrait>,
    ) {
        let x = x.as_ref();
        assert_eq!(x.len() as i32 % self.d(), 0);
        let len = x.len() / self.d() as usize;
        let distances = distances.as_mut();
        let labels = labels.as_mut();
        assert_eq!(distances.len(), len * k as usize);
        assert_eq!(labels.len(), len * k as usize);
        assert_eq!(distances.len(), labels.len());
        unsafe {
            ffi::index_search(
                self.ptr(),
                len as i64,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
                params.map(|v| v.ptr()).unwrap_or(null_mut()),
            )
        }
    }

    fn range_search(
        &self,
        x: impl AsRef<[f32]>,
        radius: f32,
        results: &mut RangeSearchResult,
        params: Option<&impl SearchParametersTrait>,
    ) {
        let x = x.as_ref();
        assert_eq!(x.len() as i32 % self.d(), 0);
        let n = x.len() / self.d() as usize;
        unsafe {
            ffi::index_range_search(
                self.ptr(),
                n as i64,
                x.as_ptr(),
                radius,
                results.ptr(),
                params.map(|v| v.ptr()).unwrap_or(null_mut()),
            )
        }
    }

    fn assign(&self, x: &[f32], labels: &mut [i64], k: i64) {
        unsafe {
            ffi::index_assign(
                self.ptr(),
                x.len() as i64,
                x.as_ptr(),
                labels.as_mut_ptr(),
                k,
            )
        }
    }

    fn reset(&mut self) {
        unsafe { ffi::index_reset(self.ptr()) }
    }

    fn remove_ids(&mut self, sel: &impl IDSelectorTrait) -> usize {
        unsafe { ffi::index_remove_ids(self.ptr(), sel.ptr()) }
    }

    fn reconstruct(&self, key: i64, recons: &mut [f32]) {
        unsafe { ffi::index_reconstruct(self.ptr(), key, recons.as_mut_ptr()) }
    }

    fn reconstruct_batch(&self, keys: &[i64], recons: &mut [f32]) {
        unsafe {
            ffi::index_reconstruct_batch(
                self.ptr(),
                keys.len() as i64,
                keys.as_ptr(),
                recons.as_mut_ptr(),
            )
        }
    }

    fn reconstruct_n(&self, i0: i64, ni: i64, recons: &mut [f32]) {
        unsafe { ffi::index_reconstruct_n(self.ptr(), i0, ni, recons.as_mut_ptr()) }
    }

    fn search_and_reconstruct(
        &mut self,
        x: &[f32],
        k: i64,
        distances: &mut [f32],
        labels: &mut [i64],
        recons: &mut [f32],
        params: Option<&impl SearchParametersTrait>,
    ) {
        unsafe {
            ffi::index_search_and_reconstruct(
                self.ptr(),
                x.len() as i64,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
                recons.as_mut_ptr(),
                params.map(|p| p.ptr()).unwrap_or(null_mut()),
            )
        }
    }

    fn compute_residual(&self, x: &[f32], residuals: &mut [f32], key: i64) {
        unsafe { ffi::index_compute_residual(self.ptr(), x.as_ptr(), residuals.as_mut_ptr(), key) }
    }

    fn compute_residual_n(&self, x: &[f32], residuals: &mut [f32], keys: &[i64]) {
        unsafe {
            ffi::index_compute_residual_n(
                self.ptr(),
                x.len() as i64,
                x.as_ptr(),
                residuals.as_mut_ptr(),
                keys.as_ptr(),
            )
        }
    }

    fn get_distance_computer(&self) -> IndexDistanceComputer<'_, Self>
    where
        Self: Sized,
    {
        let inner = unsafe { ffi::index_get_distance_computer(self.ptr()) };
        IndexDistanceComputer {
            inner,
            marker: PhantomData,
        }
    }

    fn sa_code_size(&self) -> usize {
        unsafe { ffi::index_sa_code_size(self.ptr()) }
    }

    fn sa_encode(&self, x: &[f32], bytes: &mut [u8]) {
        unsafe { ffi::index_sa_encode(self.ptr(), x.len() as i64, x.as_ptr(), bytes.as_mut_ptr()) }
    }

    fn sa_decode(&self, bytes: &[u8], x: &mut [f32]) {
        unsafe { ffi::index_sa_decode(self.ptr(), x.len() as i64, bytes.as_ptr(), x.as_mut_ptr()) }
    }

    fn merge_from(&mut self, rhs: &mut impl IndexTrait, add_id: i64) {
        unsafe { ffi::index_merge_from(self.ptr(), rhs.ptr(), add_id) }
    }

    fn check_compatible_for_merge(&self, other: &impl IndexTrait) -> bool {
        unsafe { ffi::index_check_compatible_for_merge(self.ptr(), other.ptr()) }
    }
}

pub type SearchParametersPtr = *mut i32;

pub trait SearchParametersTrait {
    fn ptr(&self) -> SearchParametersPtr;
    fn into_ptr(self) -> SearchParametersPtr;
    fn sel(&self) -> Option<&dyn IDSelectorTrait>;
    fn set_sel(&mut self, sel: impl IDSelectorTrait + 'static);
}

#[derive(smart_default::SmartDefault)]
pub struct SearchParametersImpl {
    #[default(
        unsafe { ffi::search_parameters_new(null_mut()) }
    )]
    inner: SearchParametersPtr,
    sel: Option<Box<dyn IDSelectorTrait>>,
}

impl SearchParametersImpl {
    pub fn new(sel: impl IDSelectorTrait + 'static) -> Self {
        let mut sp = Self::default();
        sp.set_sel(sel);
        sp
    }
}

impl SearchParametersTrait for SearchParametersImpl {
    fn ptr(&self) -> SearchParametersPtr {
        self.inner
    }

    fn into_ptr(self) -> SearchParametersPtr {
        let mut s = self;
        let inner = s.inner;
        s.inner = null_mut();
        inner
    }
    fn sel(&self) -> Option<&dyn IDSelectorTrait> {
        self.sel.as_deref()
    }

    fn set_sel(&mut self, sel: impl IDSelectorTrait + 'static) {
        unsafe { ffi::search_parameters_set_sel(self.ptr(), sel.ptr()) }
        self.sel = Some(Box::new(sel));
    }
}
