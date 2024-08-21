use std::mem::forget;
use std::ptr::null_mut;

use crate::{
    error::*,
    impl_aux_index_structure::FaissIDSelectorBorrowed,
    index::{FaissIndexBorrowed, FaissSearchParametersTrait},
    traits::{FaissIDSelectorTrait, FaissIndexTrait},
};
use faiss_next_sys as ffi;

pub struct FaissSearchParametersIVF {
    pub inner: *mut ffi::FaissSearchParameters,
}

impl FaissSearchParametersIVF {
    pub fn downcast(params: impl FaissSearchParametersTrait) -> Self {
        let inner = unsafe { ffi::faiss_SearchParametersIVF_cast(params.inner()) };
        forget(params);
        Self { inner }
    }
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc(unsafe { ffi::faiss_SearchParametersIVF_new(&mut inner) })?;
        Ok(Self { inner })
    }
    pub fn new_with(
        sel: impl FaissIDSelectorTrait,
        nprobe: usize,
        max_codes: usize,
    ) -> Result<Self> {
        let sel_inner = sel.inner();
        let mut inner = null_mut();
        forget(sel);
        faiss_rc(unsafe {
            ffi::faiss_SearchParametersIVF_new_with(&mut inner, sel_inner, nprobe, max_codes)
        })?;
        Ok(Self { inner })
    }

    pub fn sel(&self) -> FaissIDSelectorBorrowed<'_, Self> {
        let inner = unsafe { ffi::faiss_SearchParametersIVF_sel(self.inner) };
        FaissIDSelectorBorrowed {
            inner,
            owner: std::marker::PhantomData,
        }
    }

    pub fn nprobe(&self) -> usize {
        unsafe { ffi::faiss_SearchParametersIVF_nprobe(self.inner) }
    }
    pub fn set_nprobe(&mut self, nprobe: usize) {
        unsafe { ffi::faiss_SearchParametersIVF_set_nprobe(self.inner, nprobe) }
    }

    pub fn max_codes(&self) -> usize {
        unsafe { ffi::faiss_SearchParametersIVF_max_codes(self.inner) }
    }
    pub fn set_max_codes(&mut self, max_codes: usize) {
        unsafe { ffi::faiss_SearchParametersIVF_set_max_codes(self.inner, max_codes) }
    }
}

pub trait FaissIndexIVFTrait: FaissIndexTrait + Sized {
    fn nlist(&self) -> usize {
        unsafe { ffi::faiss_IndexIVF_nlist(self.inner()) }
    }
    fn nprobe(&self) -> usize {
        unsafe { ffi::faiss_IndexIVF_nprobe(self.inner()) }
    }
    fn set_nprobe(&mut self, nprobe: usize) {
        unsafe { ffi::faiss_IndexIVF_set_nprobe(self.inner(), nprobe) }
    }
    fn quantizer(&self) -> Option<FaissIndexBorrowed<'_, Self>> {
        let inner = unsafe { ffi::faiss_IndexIVF_quantizer(self.inner()) };
        match inner.is_null() {
            true => None,
            false => Some(FaissIndexBorrowed {
                inner,
                owner: std::marker::PhantomData,
            }),
        }
    }

    fn quantizer_train_alone(&self) -> i8 {
        unsafe { ffi::faiss_IndexIVF_quantizer_trains_alone(self.inner()) }
    }

    fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexIVF_own_fields(self.inner()) > 0 }
    }
    fn set_own_fields(&mut self, own_fields: bool) {
        unsafe { ffi::faiss_IndexIVF_set_own_fields(self.inner(), own_fields as i32) }
    }
    fn merge_from(&mut self, other: impl FaissIndexTrait, add_id: i64) -> Result<()> {
        let other_inner = other.inner();
        faiss_rc(unsafe { ffi::faiss_IndexIVF_merge_from(self.inner(), other_inner, add_id) })
    }

    fn copy_subset_to(&self, other: &Self, subset_type: i32, a1: i64, a2: i64) -> Result<()> {
        faiss_rc(unsafe {
            ffi::faiss_IndexIVF_copy_subset_to(self.inner(), other.inner(), subset_type, a1, a2)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn search_preassigned(
        &self,
        x: impl AsRef<[f32]>,
        k: i64,
        assign: impl AsRef<[i64]>,
        centroid_dis: impl AsRef<[f32]>,
        mut distances: impl AsMut<[f32]>,
        mut labels: impl AsMut<[i64]>,
        store_pairs: bool,
    ) -> Result<()> {
        let n = x.as_ref().len() as i64 / self.d() as i64;
        assert_eq!(assign.as_ref().len() as i64, n * self.nprobe() as i64);
        assert_eq!(centroid_dis.as_ref().len() as i64, n * self.nprobe() as i64);
        assert_eq!(distances.as_mut().len() as i64, n * k);
        assert_eq!(labels.as_mut().len() as i64, n * k);
        faiss_rc(unsafe {
            ffi::faiss_IndexIVF_search_preassigned(
                self.inner(),
                n,
                x.as_ref().as_ptr(),
                k,
                assign.as_ref().as_ptr(),
                centroid_dis.as_ref().as_ptr(),
                distances.as_mut().as_mut_ptr(),
                labels.as_mut().as_mut_ptr(),
                store_pairs as i32,
            )
        })
    }

    fn get_list_size(&self, list_no: usize) -> usize {
        unsafe { ffi::faiss_IndexIVF_get_list_size(self.inner(), list_no) }
    }

    fn make_direct_map(&mut self, direct_map: bool) -> Result<()> {
        faiss_rc(unsafe { ffi::faiss_IndexIVF_make_direct_map(self.inner(), direct_map as i32) })
    }

    fn imbalance_factor(&self) -> f64 {
        unsafe { ffi::faiss_IndexIVF_imbalance_factor(self.inner()) }
    }

    fn print_stats(&self) {
        unsafe { ffi::faiss_IndexIVF_print_stats(self.inner()) }
    }

    fn invlists_get_ids(&self, list_no: usize) -> i64 {
        let mut invlist = 0i64;
        unsafe { ffi::faiss_IndexIVF_invlists_get_ids(self.inner(), list_no, &mut invlist) };
        invlist
    }

    fn train_encoder(&mut self, x: impl AsRef<[f32]>, assign: impl AsRef<[i64]>) -> Result<()> {
        let n = x.as_ref().len() as i64 / self.d() as i64;
        assert_eq!(assign.as_ref().len() as i64, n * self.nprobe() as i64);
        faiss_rc(unsafe {
            ffi::faiss_IndexIVF_train_encoder(
                self.inner(),
                n,
                x.as_ref().as_ptr(),
                assign.as_ref().as_ptr(),
            )
        })
    }
}

pub use ffi::FaissIndexIVFStats;

pub fn faiss_index_ivf_stats_reset() {
    let stats = faiss_get_index_ivf_stats();
    let stats: *const _ = stats;
    unsafe { ffi::faiss_IndexIVFStats_reset(stats as *mut _) }
}

pub fn faiss_get_index_ivf_stats() -> &'static FaissIndexIVFStats {
    unsafe { ffi::faiss_get_indexIVF_stats().as_ref() }.unwrap()
}
