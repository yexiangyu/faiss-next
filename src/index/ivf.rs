use super::{Index, IndexPtr};
use crate::id_selector::IDSelector;
use crate::macros::faiss_rc;
use crate::{error::Result, macros::define_index_impl};
use faiss_next_sys as sys;
use std::marker::PhantomData;
use std::ptr::null_mut;

use super::{
    ivf_flat::{FaissQuantizer, FaissQuantizerType},
    SearchParameters, SearchParametersPtr,
};

pub struct FaissSearchParametersIVF<'a> {
    inner: *mut sys::FaissSearchParametersIVF,
    selector: std::marker::PhantomData<&'a dyn IDSelector>,
}

impl SearchParametersPtr for FaissSearchParametersIVF<'_> {
    fn ptr(&self) -> *const sys::FaissSearchParameters {
        self.inner
    }

    fn mut_ptr(&mut self) -> *mut sys::FaissSearchParameters {
        self.inner
    }

    fn into_ptr(self) -> *mut sys::FaissSearchParameters {
        todo!()
    }
}

impl SearchParameters for FaissSearchParametersIVF<'_> {}

pub struct ConstFaissIDSelector {
    inner: *const sys::FaissIDSelector,
}

impl IDSelector for ConstFaissIDSelector {
    fn ptr(&self) -> *const sys::FaissIDSelector {
        self.inner
    }

    fn mut_ptr(&mut self) -> *mut sys::FaissIDSelector {
        todo!()
    }
}

impl FaissSearchParametersIVF<'_> {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_SearchParametersIVF_new(&mut inner) })?;
        Ok(Self {
            inner,
            selector: PhantomData,
        })
    }

    pub fn new_with(sel: &mut impl IDSelector, nprobe: usize, max_codes: usize) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_SearchParametersIVF_new_with(&mut inner, sel.mut_ptr(), nprobe, max_codes)
        })?;
        Ok(Self {
            inner,
            selector: PhantomData,
        })
    }

    pub fn nprobe(&self) -> usize {
        unsafe { sys::faiss_SearchParametersIVF_nprobe(self.ptr()) }
    }

    pub fn sel(&self) -> ConstFaissIDSelector {
        let inner = unsafe { sys::faiss_SearchParametersIVF_sel(self.ptr()) };
        ConstFaissIDSelector { inner }
    }

    pub fn max_codes(&self) -> usize {
        unsafe { sys::faiss_SearchParametersIVF_max_codes(self.ptr()) }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(i32)]
pub enum FaissIndexIVFSubsetType {
    Range = 0,
    Mod = 1,
    InvertList = 2,
}

define_index_impl!(FaissIndexIVF, faiss_IndexIVF_free);

impl FaissIndexIVF {
    pub fn downcast(index: impl Index) -> Result<Self> {
        let inner = index.into_ptr();
        let inner = unsafe { sys::faiss_IndexIVF_cast(inner) };
        if inner.is_null() {
            return Err(crate::error::Error::CastFailed);
        }
        Ok(Self { inner })
    }

    pub fn nlist(&self) -> usize {
        unsafe { sys::faiss_IndexIVF_nlist(self.ptr()) }
    }

    pub fn nprobe(&self) -> usize {
        unsafe { sys::faiss_IndexIVF_nprobe(self.ptr()) }
    }

    pub fn quantizer(&self) -> FaissQuantizer {
        let inner = unsafe { sys::faiss_IndexIVF_quantizer(self.ptr()) };
        FaissQuantizer {
            inner,
            index: PhantomData,
        }
    }

    pub fn quantizer_trains_alone(&self) -> FaissQuantizerType {
        match unsafe { sys::faiss_IndexIVF_quantizer_trains_alone(self.ptr()) } {
            0 => FaissQuantizerType::UseQuantizerAsIndex,
            1 => FaissQuantizerType::PassTrainingSet,
            2 => FaissQuantizerType::TrainingOnIndex,
            _ => unimplemented!(),
        }
    }
    //TODO: wtf?
    pub fn own_fields(&self) -> bool {
        unsafe { sys::faiss_IndexIVF_own_fields(self.ptr()) != 0 }
    }

    //TODO: wtf?
    pub fn set_own_fields(&mut self, own_fields: bool) {
        unsafe { sys::faiss_IndexIVF_set_own_fields(self.mut_ptr(), own_fields as i32) }
    }

    pub fn merge_from(&mut self, other: &mut FaissIndexIVF, add_id: i64) -> Result<()> {
        faiss_rc!({ sys::faiss_IndexIVF_merge_from(self.mut_ptr(), other.mut_ptr(), add_id) })?;
        Ok(())
    }

    pub fn copy_subset_to(
        &self,
        index: &mut FaissIndexIVF,
        subset_type: FaissIndexIVFSubsetType,
        a1: i64,
        a2: i64,
    ) -> Result<()> {
        faiss_rc!({
            sys::faiss_IndexIVF_copy_subset_to(
                self.ptr(),
                index.mut_ptr(),
                subset_type as i32,
                a1,
                a2,
            )
        })?;
        Ok(())
    }

    // TODO: wtf?
    #[allow(clippy::too_many_arguments)]
    pub fn search_preassigned(
        &mut self,
        x: impl AsRef<[f32]>,
        k: i64,
        assign: impl AsRef<[i64]>,
        centroiid_dis: impl AsRef<[f32]>,
        mut distances: impl AsMut<[f32]>,
        mut labels: impl AsMut<[i64]>,
        store_pairs: bool,
    ) -> Result<()> {
        faiss_rc!({
            sys::faiss_IndexIVF_search_preassigned(
                self.mut_ptr(),
                x.as_ref().len() as i64 / self.d(),
                x.as_ref().as_ptr(),
                k,
                assign.as_ref().as_ptr(),
                centroiid_dis.as_ref().as_ptr(),
                distances.as_mut().as_mut_ptr(),
                labels.as_mut().as_mut_ptr(),
                store_pairs as i32,
            )
        })?;
        Ok(())
    }
    pub fn get_list_size(&self, list_no: usize) -> usize {
        unsafe { sys::faiss_IndexIVF_get_list_size(self.ptr(), list_no) }
    }

    pub fn make_direct_map(&mut self, new_maintain_direct_map: bool) -> Result<()> {
        faiss_rc!({
            sys::faiss_IndexIVF_make_direct_map(self.mut_ptr(), new_maintain_direct_map as i32)
        })?;
        Ok(())
    }

    pub fn imbalance_factor(&self) -> f64 {
        unsafe { sys::faiss_IndexIVF_imbalance_factor(self.ptr()) }
    }

    pub fn print_stats(&self) {
        unsafe { sys::faiss_IndexIVF_print_stats(self.ptr()) }
    }

    pub fn invlists_get_ids(&self, list_no: usize, mut invlist: impl AsMut<[i64]>) {
        unsafe {
            sys::faiss_IndexIVF_invlists_get_ids(self.ptr(), list_no, invlist.as_mut().as_mut_ptr())
        };
    }
}

pub struct FaissIndexIVFStats {
    inner: *mut sys::FaissIndexIVFStats,
}

impl Default for FaissIndexIVFStats {
    fn default() -> Self {
        Self::new()
    }
}

impl FaissIndexIVFStats {
    pub fn new() -> Self {
        let inner = unsafe { sys::faiss_get_indexIVF_stats() };
        Self { inner }
    }

    pub fn reset(&mut self) {
        unsafe { sys::faiss_IndexIVFStats_reset(self.inner) }
    }
}
