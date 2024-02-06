use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr::null_mut;

use crate::error::{Error, Result};
use crate::macros::{define_index_impl, faiss_rc};

use super::factory::FaissIndexImpl;
use super::{Index, IndexPtr};
use faiss_next_sys as sys;
use sys::FaissMetricType;

pub trait IndexFlat: Index {
    fn xb(&mut self) -> &[f32] {
        unsafe {
            let mut len = 0;
            let mut ptr = null_mut();
            sys::faiss_IndexFlat_xb(self.mut_ptr(), &mut ptr, &mut len);
            std::slice::from_raw_parts(ptr, len)
        }
    }

    fn compute_distance_subset(
        &mut self,
        x: impl AsRef<[f32]>,
        k: i64,
        labels: impl AsRef<[i64]>,
    ) -> Result<Vec<f32>> {
        let index = self.mut_ptr();
        let x = x.as_ref();
        let n = x.len() / self.d() as usize;
        let mut distances = vec![0.0f32; n];
        let labels = labels.as_ref();
        faiss_rc!({
            sys::faiss_IndexFlat_compute_distance_subset(
                index,
                n as i64,
                x.as_ptr(),
                k,
                distances.as_mut_ptr(),
                labels.as_ptr(),
            )
        })?;
        Ok(distances)
    }
}

define_index_impl!(FaissIndexFlatImpl, faiss_IndexFlat_free);

impl IndexFlat for FaissIndexFlatImpl {}

///```rust
/// use faiss_next::{index::{flat::{FaissIndexFlatImpl, IndexFlat}, Index}};
/// let mut index = FaissIndexFlatImpl::new_with(128, faiss_next::index::FaissMetricType::METRIC_L2).expect("failed to create index?");
/// index.add(&vec![1.0; 128]).expect("add failed?");
/// let (_, distances) = index.search(&vec![1.0; 128], 1).expect("failed to search");
/// assert_eq!(distances[0], 0.0);
impl FaissIndexFlatImpl {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexFlat_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn new_with(d: i64, metric: FaissMetricType) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexFlat_new_with(&mut inner, d, metric) })?;
        Ok(Self { inner })
    }

    pub fn cast(rhs: impl IndexPtr) -> Result<Self> {
        let rhs = rhs.into_ptr();
        let inner = unsafe { sys::faiss_IndexFlat_cast(rhs) };
        if inner.is_null() {
            return Err(Error::CastFailed);
        }
        Ok(Self { inner })
    }
}

define_index_impl!(
    ///```rust
    /// use faiss_next::{index::{flat::{FaissIndexFlatIP, IndexFlat}, Index}};
    ///
    /// let mut index = FaissIndexFlatIP::new_with(128).expect("failed to create index?");
    ///
    /// index.add(&vec![1.0; 128]).expect("add failed?");
    ///
    /// let (_, distances) = index.search(&vec![1.0; 128], 1).expect("failed to search");
    ///
    /// assert_eq!(distances[0], 128.0);
    FaissIndexFlatIP,
    faiss_IndexFlatIP_free
);
impl IndexFlat for FaissIndexFlatIP {}

impl FaissIndexFlatIP {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexFlatIP_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn new_with(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexFlatIP_new_with(&mut inner, d) })?;
        Ok(Self { inner })
    }

    pub fn cast(rhs: impl IndexPtr) -> Result<Self> {
        let rhs = rhs.into_ptr();
        let inner = unsafe { sys::faiss_IndexFlatIP_cast(rhs) };
        if inner.is_null() {
            return Err(Error::CastFailed);
        }
        Ok(Self { inner })
    }
}

define_index_impl!(
    ///```rust
    /// use faiss_next::{index::{flat::{FaissIndexFlatL2, IndexFlat}, Index}};
    ///
    /// let mut index = FaissIndexFlatL2::new_with(128).expect("failed to create index?");
    ///
    /// index.add(&vec![1.0; 128]).expect("add failed?");
    ///
    /// let (_, distances) = index.search(&vec![1.0; 128], 1).expect("failed to search");
    ///
    /// assert_eq!(distances[0], 0.0);
    FaissIndexFlatL2,
    faiss_IndexFlatL2_free
);
impl IndexFlat for FaissIndexFlatL2 {}

impl FaissIndexFlatL2 {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexFlatL2_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn new_with(d: i64) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexFlatL2_new_with(&mut inner, d) })?;
        Ok(Self { inner })
    }

    pub fn cast(rhs: impl IndexPtr) -> Result<Self> {
        let rhs = rhs.into_ptr();
        let inner = unsafe { sys::faiss_IndexFlatL2_cast(rhs) };
        if inner.is_null() {
            return Err(Error::CastFailed);
        }
        Ok(Self { inner })
    }
}

///```rust
/// use faiss_next::{index::{flat::{FaissIndexFlatL2, FaissIndexRefineFlat, IndexFlat}, Index}};
///
/// let mut index = FaissIndexFlatL2::new_with(128).expect("failed to create index?");
///
/// let mut refine_index = FaissIndexRefineFlat::new(&mut index).expect("failed to create refine index?");
///
/// refine_index.add(&vec![1.0; 128]).expect("add failed?");
///
/// let (_, distances) = refine_index.search(&vec![1.0; 128], 1).expect("failed to search");
///
/// assert_eq!(distances[0], 0.0);
pub struct FaissIndexRefineFlat<'a> {
    inner: *mut sys::FaissIndexRefineFlat,
    #[allow(unused)]
    index: PhantomData<&'a dyn IndexPtr>,
}

impl<'a> IndexPtr for FaissIndexRefineFlat<'a> {
    fn ptr(&self) -> *const sys::FaissIndex {
        self.inner as *const _
    }

    fn mut_ptr(&mut self) -> *mut sys::FaissIndex {
        self.inner
    }

    fn into_ptr(mut self) -> *mut sys::FaissIndex {
        let inner = self.inner;
        self.inner = null_mut();
        inner
    }
}
impl<'a> Index for FaissIndexRefineFlat<'a> {}
impl<'a> IndexFlat for FaissIndexRefineFlat<'a> {}

impl<'a> FaissIndexRefineFlat<'a> {
    pub fn new(index: &'a mut impl IndexPtr) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexRefineFlat_new(&mut inner, index.mut_ptr()) })?;
        Ok(Self {
            inner,
            index: PhantomData,
        })
    }

    pub fn cast(index: &'a mut impl IndexPtr) -> Result<Self> {
        let inner = unsafe { sys::faiss_IndexRefineFlat_cast(index.mut_ptr()) };
        if inner.is_null() {
            return Err(Error::CastFailed);
        }
        Ok(Self {
            inner,
            index: PhantomData,
        })
    }

    pub fn own_fields(&self) -> bool {
        unsafe { sys::faiss_IndexRefineFlat_own_fields(self.ptr()) != 0 }
    }

    pub fn set_own_fields(&mut self, own_fields: bool) {
        unsafe { sys::faiss_IndexRefineFlat_set_own_fields(self.mut_ptr(), own_fields as i32) }
    }

    pub fn k_factor(&self) -> f32 {
        unsafe { sys::faiss_IndexRefineFlat_k_factor(self.ptr()) }
    }
}

define_index_impl!(
    ///```rust
    /// use faiss_next::{index::{flat::{FaissIndexFlat1D, IndexFlat}, Index}};
    ///
    /// let mut index = FaissIndexFlat1D::new_with(false).expect("failed to create index?");
    ///
    /// index.add(&vec![1.0; 128 * 128]).expect("add failed?");
    ///
    /// index.update_permutation().expect("failed to update permutation");
    ///
    /// let (_, distances) = index.search(&vec![1.0; 128], 1).expect("failed to search");
    ///
    /// assert_eq!(distances[0], 0.0);
    FaissIndexFlat1D,
    faiss_IndexFlat1D_free
);
impl IndexFlat for FaissIndexFlat1D {}

impl FaissIndexFlat1D {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexFlat1D_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn new_with(continuous_update: bool) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexFlat1D_new_with(&mut inner, continuous_update as i32) })?;
        Ok(Self { inner })
    }

    pub fn cast(rhs: impl IndexPtr) -> Result<Self> {
        let rhs = rhs.into_ptr();
        let inner = unsafe { sys::faiss_IndexFlat1D_cast(rhs) };
        if inner.is_null() {
            return Err(Error::CastFailed);
        }
        Ok(Self { inner })
    }
    pub fn update_permutation(&mut self) -> Result<()> {
        faiss_rc!({ sys::faiss_IndexFlat1D_update_permutation(self.mut_ptr()) })?;
        Ok(())
    }
}

// TODO: Test
define_index_impl!(FaissIndexIVFFlat, faiss_IndexIVFFlat_free);
impl IndexFlat for FaissIndexIVFFlat {}

impl FaissIndexIVFFlat {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexIVFFlat_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn new_with(quantizer: &mut impl IndexPtr, d: usize, nlist: usize) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({ sys::faiss_IndexIVFFlat_new_with(&mut inner, quantizer.mut_ptr(), d, nlist) })?;
        Ok(Self { inner })
    }

    pub fn new_with_metric(
        quantizer: &mut impl IndexPtr,
        d: usize,
        nlist: usize,
        metric: FaissMetricType,
    ) -> Result<Self> {
        let mut inner = null_mut();
        faiss_rc!({
            sys::faiss_IndexIVFFlat_new_with_metric(
                &mut inner,
                quantizer.mut_ptr(),
                d,
                nlist,
                metric,
            )
        })?;
        Ok(Self { inner })
    }

    pub fn cast(rhs: impl IndexPtr) -> Result<Self> {
        let rhs = rhs.into_ptr();
        let inner = unsafe { sys::faiss_IndexIVFFlat_cast(rhs) };
        if inner.is_null() {
            return Err(Error::CastFailed);
        }
        Ok(Self { inner })
    }

    pub fn nlist(&self) -> usize {
        unsafe { sys::faiss_IndexIVFFlat_nlist(self.ptr()) }
    }

    pub fn nprobe(&self) -> usize {
        unsafe { sys::faiss_IndexIVFFlat_nprobe(self.ptr()) }
    }

    pub fn set_nprobe(&mut self, nprobe: usize) {
        unsafe { sys::faiss_IndexIVFFlat_set_nprobe(self.mut_ptr(), nprobe) }
    }

    pub fn quantizer(&self) -> ManuallyDrop<FaissIndexImpl> {
        let inner = unsafe { sys::faiss_IndexIVFFlat_quantizer(self.ptr()) };
        FaissIndexImpl::from_ptr(inner)
    }

    pub fn quantizer_trains_alone(&self) -> i8 {
        unsafe { sys::faiss_IndexIVFFlat_quantizer_trains_alone(self.ptr()) }
    }

    pub fn own_fields(&self) -> bool {
        unsafe { sys::faiss_IndexIVFFlat_own_fields(self.ptr()) != 0 }
    }

    pub fn set_own_fields(&mut self, own_fields: bool) {
        unsafe { sys::faiss_IndexIVFFlat_set_own_fields(self.mut_ptr(), own_fields as i32) }
    }

    pub fn add_core(
        &mut self,
        x: impl AsRef<[f32]>,
        xids: impl AsRef<[i64]>,
        precomputed_idx: impl AsRef<[i64]>,
    ) -> Result<()> {
        let x = x.as_ref();
        let xids = xids.as_ref();
        let precomputed_idx = precomputed_idx.as_ref();
        let n = x.len() as i64 / self.d();
        faiss_rc!({
            sys::faiss_IndexIVFFlat_add_core(
                self.mut_ptr(),
                n,
                x.as_ptr(),
                xids.as_ptr(),
                precomputed_idx.as_ptr(),
            )
        })?;
        Ok(())
    }

    pub fn update_vectors(&mut self, v: impl AsRef<[f32]>) -> Result<Vec<i64>> {
        let v = v.as_ref();
        let nv = v.len() as i64 / self.d();
        let mut ret = vec![0; nv as usize];
        faiss_rc!({
            sys::faiss_IndexIVFFlat_update_vectors(
                self.mut_ptr(),
                nv as i32,
                ret.as_mut_ptr(),
                v.as_ptr(),
            )
        })?;
        Ok(ret)
    }
}
