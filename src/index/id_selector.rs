use faiss_next_sys as sys;
use std::ptr::null_mut;

use crate::error::Result;
use crate::macros::rc;

pub struct FaissIDSelector {
    pub inner: *mut sys::FaissIDSelector,
    #[allow(unused)]
    reference: (Option<Box<FaissIDSelector>>, Option<Box<FaissIDSelector>>),
}

impl Drop for FaissIDSelector {
    fn drop(&mut self) {
        unsafe { sys::faiss_IDSelector_free(self.inner) }
    }
}

impl FaissIDSelector {
    pub fn batch(ids: impl AsRef<[i64]>) -> Result<FaissIDSelector> {
        let ids = ids.as_ref();
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorBatch_new(&mut inner, ids.len(), ids.as_ptr()) })?;
        Ok(FaissIDSelector {
            inner: inner as *mut _,
            reference: (None, None),
        })
    }

    pub fn range(min: i64, max: i64) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorRange_new(&mut inner, min, max) })?;
        Ok(FaissIDSelector {
            inner: inner as *mut _,
            reference: (None, None),
        })
    }

    pub fn try_not(self) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorNot_new(&mut inner, self.inner) })?;
        Ok(FaissIDSelector {
            inner: inner as *mut _,
            reference: (Some(Box::from(self)), None),
        })
    }

    pub fn try_and(self, rhs: FaissIDSelector) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorAnd_new(&mut inner, self.inner, rhs.inner) })?;
        Ok(FaissIDSelector {
            inner: inner as *mut _,
            reference: (Some(Box::from(self)), Some(Box::from(rhs))),
        })
    }

    pub fn try_or(self, rhs: FaissIDSelector) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorOr_new(&mut inner, self.inner, rhs.inner) })?;
        Ok(FaissIDSelector {
            inner: inner as *mut _,
            reference: (Some(Box::from(self)), Some(Box::from(rhs))),
        })
    }

    pub fn try_xor(self, rhs: FaissIDSelector) -> Result<Self> {
        let mut inner = null_mut();
        rc!({ sys::faiss_IDSelectorXOr_new(&mut inner, self.inner, rhs.inner) })?;
        Ok(FaissIDSelector {
            inner: inner as *mut _,
            reference: (Some(Box::from(self)), Some(Box::from(rhs))),
        })
    }
}

impl std::ops::Not for FaissIDSelector {
    type Output = FaissIDSelector;

    fn not(self) -> Self::Output {
        self.try_not().expect("failed to not selector")
    }
}

impl std::ops::BitAnd for FaissIDSelector {
    type Output = FaissIDSelector;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.try_and(rhs).expect("failed to and selectors")
    }
}

impl std::ops::BitOr for FaissIDSelector {
    type Output = FaissIDSelector;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.try_or(rhs).expect("failed to or selectors")
    }
}

impl std::ops::BitXor for FaissIDSelector {
    type Output = FaissIDSelector;

    fn bitxor(self, rhs: Self) -> Self::Output {
        self.try_xor(rhs).expect("failed to xor selectors")
    }
}

// pub trait FaissIDSelectorTrait {
//     fn to_ptr(&self) -> *const sys::FaissIDSelector;
//     fn mut_ptr(&mut self) -> *mut sys::FaissIDSelector;
//     fn into_ptr(mut self) -> *mut sys::FaissIDSelector
//     where
//         Self: Sized,
//     {
//         let inner = self.mut_ptr();
//         std::mem::forget(self);
//         inner
//     }

//     fn is_member(&self, id: i64) -> bool {
//         unsafe { sys::faiss_IDSelector_is_member(self.to_ptr(), id) != 0 }
//     }
// }

// pub struct FaissIDSelector {
//     inner: *mut sys::FaissIDSelector,
//     reference: Option<Box<dyn FaissIDSelectorTrait>>,
// }

// impl FaissIDSelector {
//     pub fn batch(ids: impl AsRef<[i64]>) -> Result<FaissIDSelector> {
//         let ids = ids.as_ref();
//         let mut inner = null_mut();
//         rc!({ sys::faiss_IDSelectorBatch_new(&mut inner, ids.len(), ids.as_ptr()) })?;
//         Ok(FaissIDSelector {
//             inner: inner as *mut _,
//             reference: None,
//         })
//     }

//     pub fn range(min: i64, max: i64) -> Result<Self> {
//         let mut inner = null_mut();
//         rc!({ sys::faiss_IDSelectorRange_new(&mut inner, min, max) })?;
//         Ok(FaissIDSelector {
//             inner: inner as *mut _,
//             reference: None,
//         })
//     }

//     pub fn not(self) -> Result<Self> {
//         let mut inner = null_mut();
//         rc!({ sys::faiss_IDSelectorNot_new(&mut inner, self.inner) })?;
//         Ok(FaissIDSelector {
//             inner: inner as *mut _,
//             reference: None,
//         })
//     }
// }

// // pub trait IDSelectorTrait {
// //     fn ptr(&self) -> *const sys::FaissIDSelector;

// //     fn mut_ptr(&mut self) -> *mut sys::FaissIDSelector;

// //     fn is_member(&self, id: i64) -> bool {
// //         unsafe { sys::faiss_IDSelector_is_member(self.ptr(), id) != 0 }
// //     }

// //     fn not(&self) -> Result<IDSelectorNot>
// //     where
// //         Self: IDSelector + Sized,
// //     {
// //         let mut inner = null_mut();
// //         faiss_rc!({ sys::faiss_IDSelectorNot_new(&mut inner, self.ptr()) })?;
// //         Ok(IDSelectorNot {
// //             inner,
// //             selector: PhantomData,
// //         })
// //     }

// //     fn and<'a, 'b>(&'a self, rhs: &'b dyn IDSelector) -> Result<IDSelectorAnd>
// //     where
// //         Self: IDSelector + Sized,
// //         'b: 'a,
// //     {
// //         let mut inner = null_mut();
// //         faiss_rc!({ sys::faiss_IDSelectorAnd_new(&mut inner, self.ptr(), rhs.ptr()) })?;
// //         Ok(IDSelectorAnd {
// //             inner,
// //             selector_a: PhantomData,
// //             selector_b: PhantomData,
// //         })
// //     }

// //     fn or<'a, 'b>(&'a self, rhs: &'b dyn IDSelector) -> Result<IDSelectorOr>
// //     where
// //         Self: IDSelector + Sized,
// //         'b: 'a,
// //     {
// //         let mut inner = null_mut();
// //         faiss_rc!({ sys::faiss_IDSelectorOr_new(&mut inner, self.ptr(), rhs.ptr()) })?;
// //         Ok(IDSelectorOr {
// //             inner,
// //             selector_a: PhantomData,
// //             selector_b: PhantomData,
// //         })
// //     }

// //     fn xor<'a, 'b>(&'a self, rhs: &'b dyn IDSelector) -> Result<IDSelectorXor>
// //     where
// //         Self: IDSelector + Sized,
// //         'b: 'a,
// //     {
// //         let mut inner = null_mut();
// //         faiss_rc!({ sys::faiss_IDSelectorXOr_new(&mut inner, self.ptr(), rhs.ptr()) })?;
// //         Ok(IDSelectorXor {
// //             inner,
// //             selector_a: PhantomData,
// //             selector_b: PhantomData,
// //         })
// //     }
// // }

// // define_id_selector_impl!(
// //     /// batch id selector
// //     FaissIDSelectorBatch,
// //     faiss_IDSelector_free
// // );

// // impl FaissIDSelectorBatch {
// //     pub fn new(ids: impl AsRef<[i64]>) -> Result<FaissIDSelectorBatch> {
// //         let ids = ids.as_ref();
// //         let mut inner = null_mut();
// //         faiss_rc!({ sys::faiss_IDSelectorBatch_new(&mut inner, ids.len(), ids.as_ptr()) })?;
// //         Ok(FaissIDSelectorBatch { inner })
// //     }

// //     pub fn nbits(&self) -> i64 {
// //         unsafe { sys::faiss_IDSelectorBatch_nbits(self.inner) as i64 }
// //     }

// //     pub fn mask(&self) -> i64 {
// //         unsafe { sys::faiss_IDSelectorBatch_mask(self.inner) }
// //     }
// // }

// // define_id_selector_impl!(
// //     /// range id selector
// //     FaissIDSelectorRange,
// //     faiss_IDSelectorRange_free
// // );

// // impl FaissIDSelectorRange {
// //     pub fn new(begin: i64, end: i64) -> Result<FaissIDSelectorRange> {
// //         let mut inner = null_mut();
// //         faiss_rc!({ sys::faiss_IDSelectorRange_new(&mut inner, begin, end) })?;
// //         Ok(FaissIDSelectorRange { inner })
// //     }
// // }

// // macro_rules! impl_id_selector_ops {
// //     ($klass: ty) => {
// //         impl IDSelector for $klass {
// //             fn ptr(&self) -> *const sys::FaissIDSelector {
// //                 self.inner as *const _
// //             }

// //             fn mut_ptr(&mut self) -> *mut sys::FaissIDSelector {
// //                 self.inner as *mut _
// //             }
// //         }
// //     };
// // }

// // pub struct IDSelectorNot<'a> {
// //     inner: *mut sys::FaissIDSelectorNot,
// //     #[allow(unused)]
// //     selector: PhantomData<&'a dyn IDSelector>,
// // }

// // impl_id_selector_ops!(IDSelectorNot<'_>);

// // pub struct IDSelectorAnd<'a, 'b> {
// //     inner: *mut sys::FaissIDSelectorAnd,
// //     #[allow(unused)]
// //     selector_a: PhantomData<&'a dyn IDSelector>,
// //     #[allow(unused)]
// //     selector_b: PhantomData<&'b dyn IDSelector>,
// // }

// // impl_id_selector_ops!(IDSelectorAnd<'_, '_>);

// // pub struct IDSelectorOr<'a, 'b> {
// //     inner: *mut sys::FaissIDSelectorOr,
// //     #[allow(unused)]
// //     selector_a: PhantomData<&'a dyn IDSelector>,
// //     #[allow(unused)]
// //     selector_b: PhantomData<&'b dyn IDSelector>,
// // }

// // impl_id_selector_ops!(IDSelectorOr<'_, '_>);

// // pub struct IDSelectorXor<'a, 'b> {
// //     inner: *mut sys::FaissIDSelectorXOr,
// //     #[allow(unused)]
// //     selector_a: PhantomData<&'a dyn IDSelector>,
// //     #[allow(unused)]
// //     selector_b: PhantomData<&'b dyn IDSelector>,
// // }

// // impl_id_selector_ops!(IDSelectorXor<'_, '_>);
