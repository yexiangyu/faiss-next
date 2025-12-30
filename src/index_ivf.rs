use crate::error::Result;
use faiss_next_sys as ffi;

use std::{mem::forget, ptr::null_mut};

use crate::impl_aux_index_structures::IDSelectorTrait;
use crate::index::IndexTrait;

#[derive(Debug)]
pub struct SearchParametersIVF {
    inner: *mut ffi::FaissSearchParameters,
}

impl SearchParametersIVF {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_SearchParametersIVF_new, &mut inner)?;
        Ok(Self { inner })
    }
    
    pub fn new_with(
        sel: impl IDSelectorTrait,
        nprobe: usize,
        max_codes: usize,
    ) -> Result<Self> {
        let sel_inner = sel.inner();
        let mut inner = null_mut();
        forget(sel);
        ffi::ok!(faiss_SearchParametersIVF_new_with, &mut inner, sel_inner, nprobe, max_codes)?;
        Ok(Self { inner })
    }

    pub fn nprobe(&self) -> usize {
        ffi::run!(faiss_SearchParametersIVF_nprobe, self.inner)
    }
    
    pub fn set_nprobe(&mut self, nprobe: usize) {
        ffi::run!(faiss_SearchParametersIVF_set_nprobe, self.inner, nprobe)
    }

    pub fn max_codes(&self) -> usize {
        ffi::run!(faiss_SearchParametersIVF_max_codes, self.inner)
    }
    
    pub fn set_max_codes(&mut self, max_codes: usize) {
        ffi::run!(faiss_SearchParametersIVF_set_max_codes, self.inner, max_codes)
    }
}

impl crate::index::SearchParametersTrait for SearchParametersIVF {
    fn inner(&self) -> *mut ffi::FaissSearchParameters {
        self.inner
    }
}

ffi::impl_drop!(SearchParametersIVF, faiss_SearchParametersIVF_free);

pub trait IndexIVFTrait: IndexTrait {
    fn nlist(&self) -> usize {
        ffi::run!(faiss_IndexIVF_nlist, self.inner() as *mut _)
    }

    fn nprobe(&self) -> usize {
        ffi::run!(faiss_IndexIVF_nprobe, self.inner() as *mut _)
    }

    fn set_nprobe(&mut self, nprobe: usize) {
        ffi::run!(faiss_IndexIVF_set_nprobe, self.inner() as *mut _, nprobe)
    }

    fn merge_from(&mut self, other: impl IndexTrait, add_id: i64) -> Result<()> {
        ffi::ok!(faiss_IndexIVF_merge_from, self.inner() as *mut _, other.inner(), add_id)?;
        Ok(())
    }

    fn copy_subset_to(
        &self,
        other: &impl IndexTrait,
        subset_type: i32,
        a1: i64,
        a2: i64,
    ) -> Result<()> {
        ffi::ok!(
            faiss_IndexIVF_copy_subset_to,
            self.inner() as *mut _,
            other.inner(),
            subset_type,
            a1,
            a2
        )?;
        Ok(())
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
        assert_eq!(
            assign.as_ref().len() as i64, 
            n * self.nprobe() as i64,
            "assign length ({}) must equal n ({}) * nprobe ({})",
            assign.as_ref().len(),
            n,
            self.nprobe()
        );
        assert_eq!(
            centroid_dis.as_ref().len() as i64, 
            n * self.nprobe() as i64,
            "centroid_dis length ({}) must equal n ({}) * nprobe ({})",
            centroid_dis.as_ref().len(),
            n,
            self.nprobe()
        );
        assert_eq!(
            distances.as_mut().len() as i64, 
            n * k,
            "distances length ({}) must equal n ({}) * k ({})",
            distances.as_mut().len(),
            n,
            k
        );
        assert_eq!(
            labels.as_mut().len() as i64, 
            n * k,
            "labels length ({}) must equal n ({}) * k ({})",
            labels.as_mut().len(),
            n,
            k
        );

        ffi::ok!(
            faiss_IndexIVF_search_preassigned,
            self.inner() as *mut _,
            n,
            x.as_ref().as_ptr(),
            k,
            assign.as_ref().as_ptr(),
            centroid_dis.as_ref().as_ptr(),
            distances.as_mut().as_mut_ptr(),
            labels.as_mut().as_mut_ptr(),
            store_pairs as i32
        )?;
        Ok(())
    }

    fn get_list_size(&self, list_no: usize) -> usize {
        ffi::run!(faiss_IndexIVF_get_list_size, self.inner() as *mut _, list_no)
    }

    fn make_direct_map(&mut self, direct_map: bool) -> Result<()> {
        ffi::ok!(faiss_IndexIVF_make_direct_map, self.inner() as *mut _, direct_map as i32)?;
        Ok(())
    }

    fn imbalance_factor(&self) -> f64 {
        ffi::run!(faiss_IndexIVF_imbalance_factor, self.inner() as *mut _)
    }

    fn print_stats(&self) {
        ffi::run!(faiss_IndexIVF_print_stats, self.inner() as *mut _)
    }

    fn train_encoder(&mut self, x: impl AsRef<[f32]>, assign: impl AsRef<[i64]>) -> Result<()> {
        let n = x.as_ref().len() as i64 / self.d() as i64;
        assert_eq!(
            assign.as_ref().len() as i64, 
            n * self.nprobe() as i64,
            "assign length ({}) must equal n ({}) * nprobe ({})",
            assign.as_ref().len(),
            n,
            self.nprobe()
        );

        ffi::ok!(
            faiss_IndexIVF_train_encoder,
            self.inner() as *mut _,
            n,
            x.as_ref().as_ptr(),
            assign.as_ref().as_ptr()
        )?;
        Ok(())
    }
}

pub use ffi::FaissIndexIVFStats;

pub fn index_ivf_stats_reset() {
    let stats = get_index_ivf_stats();
    let stats: *const _ = stats;
    unsafe { ffi::faiss_IndexIVFStats_reset(stats as *mut _) }
}

pub fn get_index_ivf_stats() -> &'static ffi::FaissIndexIVFStats {
    unsafe { ffi::faiss_get_indexIVF_stats().as_ref() }.unwrap()
}
