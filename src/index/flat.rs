use std::ptr::null_mut;

use super::traits::*;
use super::FaissMetricType;
use crate::error::{Error, Result};
use crate::macros::*;
use faiss_next_sys as sys;

declare_index!(FaissIndexFlat);
impl_index_drop!(FaissIndexFlat, faiss_IndexFlat_free);

impl_index_owned_ptr!(FaissIndexFlat);
impl_index_mut_ptr!(FaissIndexFlat);
impl_index_borrowed_ptr!(FaissIndexFlat);
impl FaissIndexConstTrait for FaissIndexFlat {}
impl FaissIndexMutTrait for FaissIndexFlat {}
impl FaissIndexOwnedTrait for FaissIndexFlat {}

impl FaissIndexFlat {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexFlat_new(&mut inner) })?;
        Ok(Self { inner })
    }

    pub fn new_with(d: i64, metric: FaissMetricType) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IndexFlat_new_with(&mut inner, d, metric) })?;
        Ok(Self { inner })
    }

    pub fn xb(&mut self) -> &mut [f32] {
        unsafe {
            let mut len = 0;
            let mut ptr = null_mut();
            sys::faiss_IndexFlat_xb(self.to_mut(), &mut ptr, &mut len);
            std::slice::from_raw_parts_mut(ptr, len)
        }
    }

    pub fn downcast(index: impl FaissIndexOwnedTrait) -> Result<Self> {
        let index = index.into_ptr();
        let inner = unsafe { sys::faiss_IndexFlat_cast(index) };
        if inner.is_null() {
            return Err(Error::CastFailed);
        }
        Ok(Self { inner })
    }
}

// use crate::macros::{define_index_impl, faiss_rc};

// // use super::factory::FaissIndexImpl;
// use super::{Index, IndexPtr};
// use faiss_next_sys as sys;
// use sys::FaissMetricType;

// pub trait IndexFlat: Index {
//     fn xb(&mut self) -> &[f32] {
//         unsafe {
//             let mut len = 0;
//             let mut ptr = null_mut();
//             sys::faiss_IndexFlat_xb(self.mut_ptr(), &mut ptr, &mut len);
//             std::slice::from_raw_parts(ptr, len)
//         }
//     }

//     fn compute_distance_subset(
//         &mut self,
//         x: impl AsRef<[f32]>,
//         k: i64,
//         labels: impl AsRef<[i64]>,
//     ) -> Result<Vec<f32>> {
//         let index = self.mut_ptr();
//         let x = x.as_ref();
//         let n = x.len() / self.d() as usize;
//         let mut distances = vec![0.0f32; n];
//         let labels = labels.as_ref();
//         faiss_rc!({
//             sys::faiss_IndexFlat_compute_distance_subset(
//                 index,
//                 n as i64,
//                 x.as_ptr(),
//                 k,
//                 distances.as_mut_ptr(),
//                 labels.as_ptr(),
//             )
//         })?;
//         Ok(distances)
//     }
// }

// define_index_impl!(FaissIndexFlatImpl, faiss_IndexFlat_free);

// impl IndexFlat for FaissIndexFlatImpl {}

// ///```rust
// /// use faiss_next::{index::{flat::{FaissIndexFlatImpl, IndexFlat}, Index}};
// /// let mut index = FaissIndexFlatImpl::new_with(128, faiss_next::index::FaissMetricType::METRIC_L2).expect("failed to create index?");
// /// index.add(&vec![1.0; 128]).expect("add failed?");
// /// let (_, distances) = index.search(&vec![1.0; 128], 1).expect("failed to search");
// /// assert_eq!(distances[0], 0.0);
// impl FaissIndexFlatImpl {
//     pub fn new() -> Result<Self> {
//         let mut inner = null_mut();
//         faiss_rc!({ sys::faiss_IndexFlat_new(&mut inner) })?;
//         Ok(Self { inner })
//     }

//     pub fn new_with(d: i64, metric: FaissMetricType) -> Result<Self> {
//         let mut inner = null_mut();
//         faiss_rc!({ sys::faiss_IndexFlat_new_with(&mut inner, d, metric) })?;
//         Ok(Self { inner })
//     }

//     pub fn cast(rhs: impl IndexPtr) -> Result<Self> {
//         let rhs = rhs.into_ptr();
//         let inner = unsafe { sys::faiss_IndexFlat_cast(rhs) };
//         if inner.is_null() {
//             return Err(Error::CastFailed);
//         }
//         Ok(Self { inner })
//     }
// }

// define_index_impl!(
//     ///```rust
//     /// use faiss_next::{index::{flat::{FaissIndexFlatIP, IndexFlat}, Index}};
//     ///
//     /// let mut index = FaissIndexFlatIP::new_with(128).expect("failed to create index?");
//     ///
//     /// index.add(&vec![1.0; 128]).expect("add failed?");
//     ///
//     /// let (_, distances) = index.search(&vec![1.0; 128], 1).expect("failed to search");
//     ///
//     /// assert_eq!(distances[0], 128.0);
//     FaissIndexFlatIP,
//     faiss_IndexFlatIP_free
// );
// impl IndexFlat for FaissIndexFlatIP {}

// impl FaissIndexFlatIP {
//     pub fn new() -> Result<Self> {
//         let mut inner = null_mut();
//         faiss_rc!({ sys::faiss_IndexFlatIP_new(&mut inner) })?;
//         Ok(Self { inner })
//     }

//     pub fn new_with(d: i64) -> Result<Self> {
//         let mut inner = null_mut();
//         faiss_rc!({ sys::faiss_IndexFlatIP_new_with(&mut inner, d) })?;
//         Ok(Self { inner })
//     }

//     pub fn cast(rhs: impl IndexPtr) -> Result<Self> {
//         let rhs = rhs.into_ptr();
//         let inner = unsafe { sys::faiss_IndexFlatIP_cast(rhs) };
//         if inner.is_null() {
//             return Err(Error::CastFailed);
//         }
//         Ok(Self { inner })
//     }
// }

// define_index_impl!(
//     ///```rust
//     /// use faiss_next::{index::{flat::{FaissIndexFlatL2, IndexFlat}, Index}};
//     ///
//     /// let mut index = FaissIndexFlatL2::new_with(128).expect("failed to create index?");
//     ///
//     /// index.add(&vec![1.0; 128]).expect("add failed?");
//     ///
//     /// let (_, distances) = index.search(&vec![1.0; 128], 1).expect("failed to search");
//     ///
//     /// assert_eq!(distances[0], 0.0);
//     FaissIndexFlatL2,
//     faiss_IndexFlatL2_free
// );
// impl IndexFlat for FaissIndexFlatL2 {}

// impl FaissIndexFlatL2 {
//     pub fn new() -> Result<Self> {
//         let mut inner = null_mut();
//         faiss_rc!({ sys::faiss_IndexFlatL2_new(&mut inner) })?;
//         Ok(Self { inner })
//     }

//     pub fn new_with(d: i64) -> Result<Self> {
//         let mut inner = null_mut();
//         faiss_rc!({ sys::faiss_IndexFlatL2_new_with(&mut inner, d) })?;
//         Ok(Self { inner })
//     }

//     pub fn cast(rhs: impl IndexPtr) -> Result<Self> {
//         let rhs = rhs.into_ptr();
//         let inner = unsafe { sys::faiss_IndexFlatL2_cast(rhs) };
//         if inner.is_null() {
//             return Err(Error::CastFailed);
//         }
//         Ok(Self { inner })
//     }
// }

// ///```rust
// /// use faiss_next::{index::{flat::{FaissIndexFlatL2, FaissIndexRefineFlat, IndexFlat}, Index}};
// ///
// /// let mut index = FaissIndexFlatL2::new_with(128).expect("failed to create index?");
// ///
// /// let mut refine_index = FaissIndexRefineFlat::new(&mut index).expect("failed to create refine index?");
// ///
// /// refine_index.add(&vec![1.0; 128]).expect("add failed?");
// ///
// /// let (_, distances) = refine_index.search(&vec![1.0; 128], 1).expect("failed to search");
// ///
// /// assert_eq!(distances[0], 0.0);
// pub struct FaissIndexRefineFlat<'a> {
//     inner: *mut sys::FaissIndexRefineFlat,
//     #[allow(unused)]
//     index: PhantomData<&'a dyn IndexPtr>,
// }

// impl Drop for FaissIndexRefineFlat<'_> {
//     fn drop(&mut self) {
//         unsafe { sys::faiss_IndexRefineFlat_free(self.mut_ptr()) }
//     }
// }

// impl IndexPtr for FaissIndexRefineFlat<'_> {
//     fn ptr(&self) -> *const sys::FaissIndex {
//         self.inner as *const _
//     }

//     fn mut_ptr(&mut self) -> *mut sys::FaissIndex {
//         self.inner
//     }

//     fn into_ptr(self) -> *mut sys::FaissIndex {
//         let inner = self.inner;
//         std::mem::forget(self);
//         inner
//     }
// }

// impl Index for FaissIndexRefineFlat<'_> {}
// impl IndexFlat for FaissIndexRefineFlat<'_> {}

// impl FaissIndexRefineFlat<'_> {
//     pub fn new(index: &mut impl IndexPtr) -> Result<Self> {
//         let mut inner = null_mut();
//         faiss_rc!({ sys::faiss_IndexRefineFlat_new(&mut inner, index.mut_ptr()) })?;
//         Ok(Self {
//             inner,
//             index: PhantomData,
//         })
//     }

//     pub fn cast(index: &mut impl IndexPtr) -> Result<Self> {
//         let inner = unsafe { sys::faiss_IndexRefineFlat_cast(index.mut_ptr()) };
//         if inner.is_null() {
//             return Err(Error::CastFailed);
//         }
//         Ok(Self {
//             inner,
//             index: PhantomData,
//         })
//     }

//     pub fn own_fields(&self) -> bool {
//         unsafe { sys::faiss_IndexRefineFlat_own_fields(self.ptr()) != 0 }
//     }

//     pub fn set_own_fields(&mut self, own_fields: bool) {
//         unsafe { sys::faiss_IndexRefineFlat_set_own_fields(self.mut_ptr(), own_fields as i32) }
//     }

//     pub fn k_factor(&self) -> f32 {
//         unsafe { sys::faiss_IndexRefineFlat_k_factor(self.ptr()) }
//     }
// }

// define_index_impl!(
//     ///```rust
//     /// use faiss_next::{index::{flat::{FaissIndexFlat1D, IndexFlat}, Index}};
//     ///
//     /// let mut index = FaissIndexFlat1D::new_with(false).expect("failed to create index?");
//     ///
//     /// index.add(&vec![1.0; 128 * 128]).expect("add failed?");
//     ///
//     /// index.update_permutation().expect("failed to update permutation");
//     ///
//     /// let (_, distances) = index.search(&vec![1.0; 128], 1).expect("failed to search");
//     ///
//     /// assert_eq!(distances[0], 0.0);
//     FaissIndexFlat1D,
//     faiss_IndexFlat1D_free
// );
// impl IndexFlat for FaissIndexFlat1D {}

// impl FaissIndexFlat1D {
//     pub fn new() -> Result<Self> {
//         let mut inner = null_mut();
//         faiss_rc!({ sys::faiss_IndexFlat1D_new(&mut inner) })?;
//         Ok(Self { inner })
//     }

//     pub fn new_with(continuous_update: bool) -> Result<Self> {
//         let mut inner = null_mut();
//         faiss_rc!({ sys::faiss_IndexFlat1D_new_with(&mut inner, continuous_update as i32) })?;
//         Ok(Self { inner })
//     }

//     pub fn cast(rhs: impl IndexPtr) -> Result<Self> {
//         let rhs = rhs.into_ptr();
//         let inner = unsafe { sys::faiss_IndexFlat1D_cast(rhs) };
//         if inner.is_null() {
//             return Err(Error::CastFailed);
//         }
//         Ok(Self { inner })
//     }
//     pub fn update_permutation(&mut self) -> Result<()> {
//         faiss_rc!({ sys::faiss_IndexFlat1D_update_permutation(self.mut_ptr()) })?;
//         Ok(())
//     }
// }
