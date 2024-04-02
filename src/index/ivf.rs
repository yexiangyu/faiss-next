use faiss_next_sys as sys;
use std::alloc::{alloc, dealloc};
use std::ffi::c_char;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::null_mut;

use crate::error::Result;
use crate::implement::id_selector::IDSelectorTrait;
use crate::index::{IndexTrait, SearchParametersTrait};
use crate::macros::rc;

pub struct SearchParametersIVF {
    inner: *mut sys::FaissSearchParametersIVF,
}

impl SearchParametersTrait for SearchParametersIVF {
    fn ptr(&self) -> *mut sys::FaissSearchParameters {
        self.inner
    }
}

impl Drop for SearchParametersIVF {
    fn drop(&mut self) {
        unsafe { sys::faiss_SearchParametersIVF_free(self.inner) }
    }
}

#[derive(Debug)]
pub struct IDSelectorBorrowed<'a> {
    inner: *mut sys::FaissIDSelector,
    _marker: PhantomData<&'a SearchParametersIVF>,
}

unsafe impl Send for IDSelectorBorrowed<'_> {}
unsafe impl Sync for IDSelectorBorrowed<'_> {}

impl IDSelectorTrait for IDSelectorBorrowed<'_> {
    fn ptr(&self) -> *mut sys::FaissIDSelector {
        self.inner
    }
}

impl SearchParametersIVF {
    pub fn new(nprobe: usize, max_codes: usize, sel: impl IDSelectorTrait) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_SearchParametersIVF_new_with(&mut inner, sel.ptr(), nprobe, max_codes) })?;
        Ok(Self { inner })
    }

    pub fn sel(&self) -> IDSelectorBorrowed<'_> {
        IDSelectorBorrowed {
            inner: unsafe { sys::faiss_SearchParametersIVF_sel(self.ptr()) as *mut _ },
            _marker: PhantomData,
        }
    }

    pub fn nprobe(&self) -> usize {
        unsafe { sys::faiss_SearchParametersIVF_nprobe(self.ptr()) }
    }

    pub fn max_codes(&self) -> usize {
        unsafe { sys::faiss_SearchParametersIVF_max_codes(self.ptr()) }
    }
}

pub struct QuantizerBorrowed<'a, T>
where
    T: IndexIVFTrait,
{
    inner: *mut sys::FaissIndex,
    _marker: PhantomData<&'a T>,
}

unsafe impl<T> Send for QuantizerBorrowed<'_, T> where T: IndexIVFTrait {}
unsafe impl<T> Sync for QuantizerBorrowed<'_, T> where T: IndexIVFTrait {}

impl<T> IndexTrait for QuantizerBorrowed<'_, T>
where
    T: IndexIVFTrait,
{
    fn ptr(&self) -> *mut sys::FaissIndex {
        self.inner
    }
}

impl<T> std::fmt::Debug for QuantizerBorrowed<'_, T>
where
    T: IndexIVFTrait,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizerBorrowed")
            .field("inner", &self.inner)
            .field("d", &self.d())
            .field("is_trained", &self.is_trained())
            .field("ntotal", &self.ntotal())
            .field("metric_type", &self.metric_type())
            .field("verbose", &self.verbose())
            .finish()
    }
}

pub trait IndexIVFTrait: IndexTrait {
    fn nlist(&self) -> usize {
        unsafe { sys::faiss_IndexIVF_nlist(self.ptr()) }
    }

    fn nprobe(&self) -> usize {
        unsafe { sys::faiss_IndexIVF_nprobe(self.ptr()) }
    }

    fn quantizer(&self) -> QuantizerBorrowed<'_, Self>
    where
        Self: Sized,
    {
        let inner = unsafe { sys::faiss_IndexIVF_quantizer(self.ptr()) };
        QuantizerBorrowed {
            inner,
            _marker: PhantomData,
        }
    }

    fn quantizer_trains_alone(&self) -> c_char {
        unsafe { sys::faiss_IndexIVF_quantizer_trains_alone(self.ptr()) }
    }

    fn own_fields(&self) -> bool {
        unsafe { sys::faiss_IndexIVF_own_fields(self.ptr()) != 0 }
    }

    fn set_own_fields(&mut self, own: bool) {
        unsafe { sys::faiss_IndexIVF_set_own_fields(self.ptr(), own as i32) }
    }

    fn merge_from(&mut self, other: &impl IndexIVFTrait, add_id: i64) -> Result<()> {
        rc!({ sys::faiss_IndexIVF_merge_from(self.ptr(), other.ptr(), add_id) })
    }

    fn copy_subset_to(
        &self,
        other: &mut impl IndexIVFTrait,
        subset_type: i32,
        a1: i64,
        a2: i64,
    ) -> Result<()> {
        rc!({ sys::faiss_IndexIVF_copy_subset_to(self.ptr(), other.ptr(), subset_type, a1, a2) })
    }

    #[allow(clippy::too_many_arguments)]
    fn search_preassigned(
        &self,
        x: impl AsRef<[f32]>,
        k: i64,
        assign: impl AsRef<[i64]>,
        centroids_dis: impl AsRef<[f32]>,
        mut distances: impl AsMut<[f32]>,
        mut labels: impl AsMut<[i64]>,
        store_pairs: bool,
    ) -> Result<()> {
        let n = x.as_ref().len() as i64 / self.d() as i64;
        rc!({
            sys::faiss_IndexIVF_search_preassigned(
                self.ptr(),
                n,
                x.as_ref().as_ptr(),
                k,
                assign.as_ref().as_ptr(),
                centroids_dis.as_ref().as_ptr(),
                distances.as_mut().as_mut_ptr(),
                labels.as_mut().as_mut_ptr(),
                store_pairs as i32,
            )
        })
    }

    fn get_list_size(&self, list_no: usize) -> usize {
        unsafe { sys::faiss_IndexIVF_get_list_size(self.ptr(), list_no) }
    }

    fn make_direct_map(&mut self, new_maintain_direct_map: bool) -> Result<()> {
        rc!({ sys::faiss_IndexIVF_make_direct_map(self.ptr(), new_maintain_direct_map as i32) })
    }

    fn imbalance_factor(&self) -> f64 {
        unsafe { sys::faiss_IndexIVF_imbalance_factor(self.ptr()) }
    }

    fn print_stats(&self) {
        unsafe { sys::faiss_IndexIVF_print_stats(self.ptr()) }
    }

    fn invlists_get_ids(&self, list_no: usize, mut invlist: impl AsMut<[i64]>) {
        let invlist = invlist.as_mut();
        unsafe { sys::faiss_IndexIVF_invlists_get_ids(self.ptr(), list_no, invlist.as_mut_ptr()) };
    }
}

pub struct IndexIVFStats {
    inner: *mut sys::FaissIndexIVFStats,
}

impl Drop for IndexIVFStats {
    fn drop(&mut self) {
        let l = std::alloc::Layout::new::<sys::FaissIndexIVFStats>();
        unsafe { dealloc(self.inner as *mut _, l) };
    }
}

impl Deref for IndexIVFStats {
    type Target = sys::FaissIndexIVFStats;

    fn deref(&self) -> &Self::Target {
        unsafe { self.inner.as_ref().unwrap() }
    }
}

impl DerefMut for IndexIVFStats {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.inner.as_mut().unwrap() }
    }
}

impl Default for IndexIVFStats {
    fn default() -> Self {
        let inner = unsafe {
            let l = std::alloc::Layout::new::<sys::FaissIndexIVFStats>();
            alloc(l) as *mut sys::FaissIndexIVFStats
        };
        let mut r = Self { inner };
        r.reset();
        r
    }
}

impl IndexIVFStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        unsafe { sys::faiss_IndexIVFStats_reset(self.inner) }
    }
}

pub struct GlobalIndexIVFStats {
    inner: *mut sys::FaissIndexIVFStats,
}

impl Default for GlobalIndexIVFStats {
    fn default() -> Self {
        let inner = unsafe { sys::faiss_get_indexIVF_stats() };
        Self { inner }
    }
}

impl Deref for GlobalIndexIVFStats {
    type Target = sys::FaissIndexIVFStats;

    fn deref(&self) -> &Self::Target {
        unsafe { self.inner.as_ref().unwrap() }
    }
}

impl DerefMut for GlobalIndexIVFStats {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.inner.as_mut().unwrap() }
    }
}
