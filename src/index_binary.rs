use crate::{
    aux_index_structures::RangeSearchResult,
    error::{Error, Result},
    id_selector::IDSelectorTrait,
    index::SearchParametersTrait,
    metric::MetricType,
};
use std::ptr::null_mut;

#[cxx::bridge]
#[allow(clippy::missing_safety_doc)]
pub mod ffi {
    unsafe extern "C++" {
        include!("faiss-next/src/cpp/index_binary.hpp");
        unsafe fn index_binary_free(ptr: *mut i32);
        unsafe fn index_binary_d(ptr: *const i32) -> i32;
        unsafe fn index_binary_code_size(ptr: *const i32) -> i32;
        unsafe fn index_binary_verbose(ptr: *const i32) -> bool;
        unsafe fn index_binary_set_verbose(ptr: *mut i32, verbose: bool);
        unsafe fn index_binary_is_trained(ptr: *const i32) -> bool;
        unsafe fn index_binary_metric_type(ptr: *const i32) -> i32;
        unsafe fn index_binary_train(ptr: *mut i32, n: i64, x: *const u8);
        unsafe fn index_binary_add(ptr: *mut i32, n: i64, x: *const u8);
        unsafe fn index_binary_add_with_ids(ptr: *mut i32, n: i64, x: *const u8, xids: *const i64);
        unsafe fn index_binary_search(
            ptr: *const i32,
            n: i64,
            x: *const u8,
            k: i64,
            distances: *mut i32,
            labels: *mut i64,
            params: *const i32,
        );
        unsafe fn index_binary_range_search(
            ptr: *const i32,
            n: i64,
            x: *const u8,
            radius: i32,
            result: *mut i32,
            params: *const i32,
        );
        unsafe fn index_binary_assign(
            ptr: *const i32,
            n: i64,
            x: *const u8,
            labels: *mut i64,
            k: i64,
        );
        unsafe fn index_binary_remove_ids(ptr: *mut i32, sel: *const i32) -> usize;
        unsafe fn index_binary_reconstruct(ptr: *const i32, key: i64, recons: *mut u8);
        unsafe fn index_binary_reconstruct_n(ptr: *const i32, i0: i64, ni: i64, recons: *mut u8);
        #[allow(clippy::too_many_arguments)]
        unsafe fn index_binary_search_and_reconstruct(
            ptr: *const i32,
            n: i64,
            x: *const u8,
            k: i64,
            distances: *mut i32,
            labels: *mut i64,
            recons: *mut u8,
            params: *const i32,
        );
        unsafe fn index_binary_display(ptr: *const i32);
        unsafe fn index_binary_merge_from(ptr: *mut i32, other: *mut i32, add_id: i64);
        unsafe fn index_binary_check_compatible_for_merge(
            ptr: *const i32,
            other: *const i32,
        ) -> bool;
    }
}

macro_rules! impl_index_binary_drop {
    ($cls: ty) => {
        impl Drop for $cls {
            fn drop(&mut self) {
                if self.inner.is_null() {
                    tracing::trace!("drop index_binary inner={:?}", self.inner);
                    unsafe { crate::index_binary::ffi::index_binary_free(self.inner) };
                }
            }
        }
    };

    ($cls: ty, $free: path) => {
        impl Drop for $cls {
            fn drop(&mut self) {
                if self.inner.is_null() {
                    tracing::trace("drop index_binary inner={:?}", self.inner)
                    unsafe { $free(self.inner) };
                }
            }
        }
    };
}

pub(crate) use impl_index_binary_drop;

macro_rules! impl_index_binary_trait {
    ($cls: ty) => {
        impl crate::index::IndexTrait for $cls {
            fn ptr(&self) -> crate::index_binary::IndexBinaryPtr {
                self.inner
            }

            fn into_ptr(self) -> crate::index_binary::IndexBinaryPtr {
                let mut s = self;
                let inner = s.inner;
                s.inner = std::ptr::null_mut();
                inner
            }
        }
    };
}

pub(crate) use impl_index_binary_trait;

pub type IndexBinaryPtr = *mut i32;

pub trait IndexBinaryTrait {
    fn ptr(&self) -> IndexBinaryPtr;
    fn into_ptr(self) -> IndexBinaryPtr;
    fn d(&self) -> i32 {
        unsafe { ffi::index_binary_d(self.ptr()) }
    }
    fn code_size(&self) -> i32 {
        unsafe { ffi::index_binary_code_size(self.ptr()) }
    }
    fn verbose(&self) -> bool {
        unsafe { ffi::index_binary_verbose(self.ptr()) }
    }
    fn set_verbose(&mut self, verbose: bool) {
        unsafe { ffi::index_binary_set_verbose(self.ptr(), verbose) }
    }
    fn is_trained(&self) -> bool {
        unsafe { ffi::index_binary_is_trained(self.ptr()) }
    }

    fn metric_type(&self) -> Result<MetricType> {
        let typ = unsafe { ffi::index_binary_metric_type(self.ptr()) };
        MetricType::try_from(typ).map_err(|_| Error::MetricFromNumber(typ))
    }
    fn train(&mut self, x: impl AsRef<[u8]>) {
        let x = x.as_ref();
        unsafe { ffi::index_binary_train(self.ptr(), x.len() as i64, x.as_ptr()) }
    }
    fn add(&mut self, x: impl AsRef<[u8]>) {
        let x = x.as_ref();
        unsafe { ffi::index_binary_add(self.ptr(), 1, x.as_ptr()) }
    }

    fn add_with_ids(&mut self, x: impl AsRef<[u8]>, xids: impl AsRef<[i64]>) {
        let x = x.as_ref();
        let xids = xids.as_ref();
        unsafe { ffi::index_binary_add_with_ids(self.ptr(), 1, x.as_ptr(), xids.as_ptr()) }
    }

    fn search(
        &self,
        x: impl AsRef<[u8]>,
        k: i64,
        mut distances: impl AsMut<[i32]>,
        mut labels: impl AsMut<[i64]>,
        params: Option<&impl SearchParametersTrait>,
    ) {
        let x = x.as_ref();
        let distances = distances.as_mut();
        let labels = labels.as_mut();
        unsafe {
            ffi::index_binary_search(
                self.ptr(),
                1,
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
        x: impl AsRef<[u8]>,
        radius: i32,
        result: &mut RangeSearchResult,
        params: Option<&impl SearchParametersTrait>,
    ) {
        let x = x.as_ref();

        unsafe {
            ffi::index_binary_range_search(
                self.ptr(),
                1,
                x.as_ptr(),
                radius,
                result.ptr(),
                params.map(|v| v.ptr()).unwrap_or(null_mut()),
            )
        }
    }

    fn assign(&self, x: impl AsRef<[u8]>, k: i64, mut labels: impl AsMut<[i64]>) {
        let x = x.as_ref();
        let labels = labels.as_mut();
        unsafe { ffi::index_binary_assign(self.ptr(), 1, x.as_ptr(), labels.as_mut_ptr(), k) }
    }

    fn remove_ids(&mut self, sel: &impl IDSelectorTrait) -> usize {
        unsafe { ffi::index_binary_remove_ids(self.ptr(), sel.ptr()) }
    }

    fn reconstruct(&self, key: i64, mut recons: impl AsMut<[u8]>) {
        let recons = recons.as_mut();
        unsafe { ffi::index_binary_reconstruct(self.ptr(), key, recons.as_mut_ptr()) }
    }

    fn reconstruct_n(&self, i0: i64, ni: i64, mut recons: impl AsMut<[u8]>) {
        let recons = recons.as_mut();
        unsafe { ffi::index_binary_reconstruct_n(self.ptr(), i0, ni, recons.as_mut_ptr()) }
    }

    fn search_and_reconstruct(
        &self,
        x: impl AsRef<[u8]>,
        k: i64,
        mut distances: impl AsMut<[i32]>,
        mut labels: impl AsMut<[i64]>,
        mut recons: impl AsMut<[u8]>,
        params: Option<&impl SearchParametersTrait>,
    ) {
        let x = x.as_ref();
        let distances = distances.as_mut();
        let labels = labels.as_mut();
        let recons = recons.as_mut();
        unsafe {
            ffi::index_binary_search_and_reconstruct(
                self.ptr(),
                1,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_mut_ptr(),
                recons.as_mut_ptr(),
                params.map(|v| v.ptr()).unwrap_or(null_mut()),
            )
        }
    }
    fn display(&self) {
        unsafe { ffi::index_binary_display(self.ptr()) }
    }

    fn merge_from(&mut self, other: &mut impl IndexBinaryTrait, add_id: i64) {
        unsafe { ffi::index_binary_merge_from(self.ptr(), other.ptr(), add_id) }
    }

    fn check_compatible_for_merge(&self, other: &impl IndexBinaryTrait) -> bool {
        unsafe { ffi::index_binary_check_compatible_for_merge(self.ptr(), other.ptr()) }
    }
}
