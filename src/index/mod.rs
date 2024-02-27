use faiss_next_sys as sys;
pub use sys::FaissMetricType;

pub mod factory;
pub mod flat;
pub mod id_selector;
pub mod owned;
pub mod parameters;
pub mod traits;

// use std::marker::PhantomData;
// use std::ptr::null_mut;

// use faiss_next_sys as sys;
// pub mod binary;
// pub mod factory;
// pub mod flat;
// pub mod ivf;
// pub mod ivf_flat;
// pub mod lsh;
// pub mod pretransform;
// pub mod replicas;

// use crate::error::Result;
// use crate::id_selector::IDSelector;
// use crate::macros::faiss_rc;
// pub use sys::FaissMetricType;

// pub trait IndexPtr {
//     fn ptr(&self) -> *const sys::FaissIndex;
//     fn mut_ptr(&mut self) -> *mut sys::FaissIndex;
//     fn into_ptr(self) -> *mut sys::FaissIndex;
// }

// pub trait Index: IndexPtr {
//     fn d(&self) -> i64 {
//         unsafe { sys::faiss_Index_d(self.ptr()) as i64 }
//     }

//     fn is_trained(&self) -> bool {
//         unsafe { sys::faiss_Index_is_trained(self.ptr()) != 0 }
//     }

//     fn ntotal(&self) -> i64 {
//         unsafe { sys::faiss_Index_ntotal(self.ptr()) }
//     }

//     fn metric_type(&self) -> FaissMetricType {
//         unsafe { sys::faiss_Index_metric_type(self.ptr()) }
//     }

//     fn verbose(&self) -> bool {
//         unsafe { sys::faiss_Index_verbose(self.ptr()) != 0 }
//     }

//     fn set_verbose(&mut self, verbose: bool) {
//         unsafe { sys::faiss_Index_set_verbose(self.mut_ptr(), verbose as i32) }
//     }

//     fn train(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
//         let x = x.as_ref();
//         let n = x.len() as i64 / self.d();
//         faiss_rc!({ sys::faiss_Index_train(self.mut_ptr(), n, x.as_ptr()) })?;
//         Ok(())
//     }

//     fn add(&mut self, x: impl AsRef<[f32]>) -> Result<()> {
//         let x = x.as_ref();
//         let n = x.len() as i64 / self.d();
//         faiss_rc!({ sys::faiss_Index_add(self.mut_ptr(), n, x.as_ptr()) })?;
//         Ok(())
//     }

//     fn add_with_ids(&mut self, x: impl AsRef<[f32]>, ids: impl AsRef<[i64]>) {
//         let x = x.as_ref();
//         let n = x.len() as i64 / self.d();
//         let ids = ids.as_ref();
//         unsafe {
//             sys::faiss_Index_add_with_ids(self.mut_ptr(), n, x.as_ptr(), ids.as_ptr());
//         }
//     }

//     fn search(&self, x: impl AsRef<[f32]>, k: i64) -> Result<(Vec<i64>, Vec<f32>)> {
//         let x = x.as_ref();
//         let n = x.len() as i64 / self.d();
//         let mut distances = vec![0.0; n as usize * k as usize];
//         let mut labels = vec![0; n as usize * k as usize];
//         faiss_rc!({
//             sys::faiss_Index_search(
//                 self.ptr(),
//                 n,
//                 x.as_ptr(),
//                 k,
//                 distances.as_mut_ptr(),
//                 labels.as_mut_ptr(),
//             )
//         })?;
//         Ok((labels, distances))
//     }

//     fn search_with_params(
//         &self,
//         x: impl AsRef<[f32]>,
//         k: i64,
//         params: impl SearchParametersPtr,
//     ) -> (Vec<i64>, Vec<f32>) {
//         let x = x.as_ref();
//         let n = x.len() as i64 / self.d();
//         let mut distances = vec![0.0; n as usize * k as usize];
//         let mut labels = vec![0; n as usize * k as usize];
//         unsafe {
//             sys::faiss_Index_search_with_params(
//                 self.ptr(),
//                 n,
//                 x.as_ptr(),
//                 k,
//                 params.ptr(),
//                 distances.as_mut_ptr(),
//                 labels.as_mut_ptr(),
//             );
//         }
//         (labels, distances)
//     }

//     //TODO: implement range search
//     fn range_search(&mut self, x: impl AsRef<[f32]>, radius: f32) -> Result<()> {
//         let x = x.as_ref();
//         let n = x.len() as i64 / self.d();
//         let mut result = FaissRangeSearchResult::new(n)?;
//         result.do_allocation()?;
//         faiss_rc!({
//             sys::faiss_Index_range_search(self.mut_ptr(), n, x.as_ptr(), radius, result.inner)
//         })?;
//         Ok(())
//     }
// }

// pub trait SearchParametersPtr {
//     fn ptr(&self) -> *const sys::FaissSearchParameters;
//     fn mut_ptr(&mut self) -> *mut sys::FaissSearchParameters;
//     fn into_ptr(self) -> *mut sys::FaissSearchParameters;
// }

// pub trait SearchParameters: SearchParametersPtr {}

// pub struct FaissSearchParametersImpl<'a> {
//     pub inner: *mut sys::FaissSearchParameters,
//     selector: PhantomData<&'a dyn IDSelector>,
// }

// impl Drop for FaissSearchParametersImpl<'_> {
//     fn drop(&mut self) {
//         unsafe { sys::faiss_SearchParameters_free(self.inner) }
//     }
// }

// impl SearchParametersPtr for FaissSearchParametersImpl<'_> {
//     fn ptr(&self) -> *const sys::FaissSearchParameters {
//         self.inner
//     }

//     fn mut_ptr(&mut self) -> *mut sys::FaissSearchParameters {
//         self.inner
//     }

//     fn into_ptr(self) -> *mut sys::FaissSearchParameters {
//         let inner = self.inner;
//         std::mem::forget(self);
//         inner
//     }
// }

// impl SearchParameters for FaissSearchParametersImpl<'_> {}

// impl FaissSearchParametersImpl<'_> {
//     pub fn new(sel: &mut impl IDSelector) -> Result<Self> {
//         let mut inner = null_mut();
//         faiss_rc!({ sys::faiss_SearchParameters_new(&mut inner, sel.mut_ptr()) })?;
//         Ok(Self {
//             inner,
//             selector: PhantomData,
//         })
//     }
// }

// pub struct FaissRangeSearchResult {
//     inner: *mut sys::FaissRangeSearchResult,
// }

// impl FaissRangeSearchResult {
//     pub fn new(nq: i64) -> Result<Self> {
//         let mut inner = null_mut();
//         faiss_rc!({ sys::faiss_RangeSearchResult_new(&mut inner, nq) })?;
//         Ok(Self { inner })
//     }

//     pub fn new_with(nq: i64, alloc_lims: i32) -> Result<Self> {
//         let mut inner = null_mut();
//         faiss_rc!({ sys::faiss_RangeSearchResult_new_with(&mut inner, nq, alloc_lims) })?;
//         Ok(Self { inner })
//     }

//     pub fn do_allocation(&mut self) -> Result<()> {
//         faiss_rc!({ sys::faiss_RangeSearchResult_do_allocation(self.inner) })?;
//         Ok(())
//     }

//     pub fn buffer_size(&self) -> usize {
//         unsafe { sys::faiss_RangeSearchResult_buffer_size(self.inner) }
//     }

//     pub fn nq(&self) -> usize {
//         unsafe { sys::faiss_RangeSearchResult_nq(self.inner) }
//     }
// }
