use std::marker::PhantomData;
use std::ptr::null_mut;

use crate::error::{Error, Result};
use crate::rc;
use faiss_next_sys as sys;

use super::common::{impl_index_drop, impl_index_trait, FaissIndexTrait};
use super::id_selector::FaissIDSelectorTrait;
use super::parameters::FaissSearchParametersTrait;

pub struct FaissSearchParametersIVF {
    inner: *mut sys::FaissSearchParametersIVF,
}

impl Drop for FaissSearchParametersIVF {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe { sys::faiss_SearchParametersIVF_free(self.inner) }
        }
    }
}

impl FaissSearchParametersTrait for FaissSearchParametersIVF {
    fn inner(&self) -> *mut sys::FaissSearchParameters {
        self.inner
    }

    fn into_inner(self) -> *mut sys::FaissSearchParameters {
        let mut s = self;
        let inner = s.inner;
        s.inner = null_mut();
        inner
    }
}

pub struct FaissIDSelectorBorrowed<'a> {
    inner: *const sys::FaissIDSelector,
    marker: PhantomData<&'a FaissSearchParametersIVF>,
}

impl FaissIDSelectorTrait for FaissIDSelectorBorrowed<'_> {
    fn is_member(&self, id: i64) -> bool {
        unsafe { sys::faiss_IDSelector_is_member(self.inner(), id) != 0 }
    }

    fn not(self) -> Result<super::id_selector::FaissIDSelectorNot>
    where
        Self: Sized + 'static,
    {
        unimplemented!()
    }

    fn and(
        self,
        _rhs: impl FaissIDSelectorTrait + 'static,
    ) -> Result<super::id_selector::FaissIDSelectorAnd>
    where
        Self: Sized + 'static,
    {
        unimplemented!()
    }

    fn or(
        self,
        _rhs: impl FaissIDSelectorTrait + 'static,
    ) -> Result<super::id_selector::FaissIDSelectorOr>
    where
        Self: Sized + 'static,
    {
        unimplemented!()
    }

    fn inner(&self) -> *mut sys::FaissIDSelector {
        self.inner as *mut _
    }
}

impl FaissSearchParametersIVF {
    pub fn new() -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        rc!({ sys::faiss_SearchParametersIVF_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn new_with(
        id_selector: impl FaissIDSelectorTrait,
        nprobe: usize,
        max_codes: usize,
    ) -> Result<Self> {
        let mut inner = std::ptr::null_mut();
        rc!({
            sys::faiss_SearchParametersIVF_new_with(
                &mut inner,
                id_selector.inner(),
                nprobe,
                max_codes,
            )
        })?;
        Ok(Self { inner })
    }

    pub fn nprobe(&self) -> usize {
        unsafe { sys::faiss_SearchParametersIVF_nprobe(self.inner) }
    }

    pub fn max_codes(&self) -> usize {
        unsafe { sys::faiss_SearchParametersIVF_max_codes(self.inner) }
    }

    pub fn sel(&self) -> FaissIDSelectorBorrowed<'_> {
        let inner = unsafe { sys::faiss_SearchParametersIVF_sel(self.inner) };
        FaissIDSelectorBorrowed {
            inner,
            marker: PhantomData,
        }
    }
}

pub trait FaissIndexIVFTrait: FaissIndexTrait {
    fn nlist(&self) -> usize {
        unsafe { sys::faiss_IndexIVF_nlist(self.inner()) }
    }

    fn nprobe(&self) -> usize {
        unsafe { sys::faiss_IndexIVF_nprobe(self.inner()) }
    }

    fn set_nprobe(&mut self, nprobe: usize) {
        unsafe { sys::faiss_IndexIVF_set_nprobe(self.inner(), nprobe) }
    }

    fn quantizer(&self) -> FaissIndexIVFQuantizer<Self>
    where
        Self: Sized,
    {
        let inner = unsafe { sys::faiss_IndexIVF_quantizer(self.inner()) };
        FaissIndexIVFQuantizer {
            inner,
            marker: PhantomData,
        }
    }

    fn quantizer_trains_alone(&self) -> i8 {
        unsafe { sys::faiss_IndexIVF_quantizer_trains_alone(self.inner()) }
    }

    fn merge_from(&mut self, rhs: impl FaissIndexIVFTrait, add_id: i64) -> Result<()> {
        rc!({ sys::faiss_IndexIVF_merge_from(self.inner(), rhs.inner(), add_id) })?;
        Ok(())
    }

    fn copy_subset_to(
        &self,
        rhs: &mut impl FaissIndexIVFTrait,
        subset_type: i32,
        a1: i64,
        a2: i64,
    ) -> Result<()> {
        rc!({
            sys::faiss_IndexIVF_copy_subset_to(self.inner(), rhs.inner(), subset_type, a1, a2)
        })?;
        Ok(())
    }

    fn search_preassigned(&self) -> Result<()> {
        // TODO
        todo!()
    }

    fn get_list_size(&self, list_no: usize) -> usize {
        unsafe { sys::faiss_IndexIVF_get_list_size(self.inner(), list_no) }
    }

    fn make_direct_map(&mut self, new: bool) -> Result<()> {
        rc!({ sys::faiss_IndexIVF_make_direct_map(self.inner(), new as i32) })?;
        Ok(())
    }

    fn imbalance_factor(&self) -> f64 {
        unsafe { sys::faiss_IndexIVF_imbalance_factor(self.inner()) }
    }

    fn print_stats(&self) {
        unsafe { sys::faiss_IndexIVF_print_stats(self.inner()) }
    }

    fn invlists_get_ids(&self, list_no: usize, invlist: &mut [i64]) {
        unsafe {
            sys::faiss_IndexIVF_invlists_get_ids(self.inner(), list_no, invlist.as_mut_ptr())
        };
    }
}

pub struct FaissIndexIVFImpl {
    inner: *mut sys::FaissIndexIVF,
}

impl_index_drop!(FaissIndexIVFImpl, faiss_IndexIVF_free);
impl_index_trait!(FaissIndexIVFImpl);

impl FaissIndexIVFImpl {
    pub fn downcast(index: impl FaissIndexTrait) -> Result<Self> {
        let i = index.into_inner();
        let i = unsafe { sys::faiss_IndexIVF_cast(i) };
        match i.is_null() {
            true => Err(Error::DowncastFailure),
            false => Ok(Self { inner: i }),
        }
    }
}

impl FaissIndexIVFTrait for FaissIndexIVFImpl {}

pub struct FaissIndexIVFStats {
    inner: *mut sys::FaissIndexIVFStats,
}

impl FaissIndexIVFStats {
    pub fn get() -> Self {
        let inner = unsafe { sys::faiss_get_indexIVF_stats() };
        Self { inner }
    }

    pub fn reset(&mut self) {
        unsafe { sys::faiss_IndexIVFStats_reset(self.inner) }
    }
}

#[allow(unused)]
pub struct FaissIndexIVFQuantizer<'a, T>
where
    T: FaissIndexIVFTrait,
{
    inner: *mut sys::FaissIndex,
    marker: PhantomData<&'a T>,
}

impl<'a, T> FaissIndexTrait for FaissIndexIVFQuantizer<'a, T>
where
    T: FaissIndexIVFTrait,
{
    fn inner(&self) -> *mut sys::FaissIndex {
        self.inner as *mut _
    }

    fn into_inner(self) -> *mut sys::FaissIndex {
        unimplemented!()
    }
}
