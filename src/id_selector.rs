#[cxx::bridge]
#[allow(clippy::missing_safety_doc)]
pub mod ffi {
    unsafe extern "C++" {
        include!("faiss-next/src/cpp/id_selector.hpp");
        unsafe fn id_selector_is_member(id_selector: *const i32, id: i64) -> bool;
        unsafe fn id_selector_free(id_selector: *mut i32);
        unsafe fn id_selector_range_new(imin: i64, imax: i64, assume_sorted: bool) -> *mut i32;
        unsafe fn id_selector_range_imin(id_selector: *const i32) -> i64;
        unsafe fn id_selector_range_imax(id_selector: *const i32) -> i64;
        unsafe fn id_selector_range_assume_sorted(id_selector: *const i32) -> bool;
        unsafe fn id_selector_array_new(n: i64, ids: *const i64) -> *mut i32;
        unsafe fn id_selector_array_ids(id_selector: *const i32) -> *const i64;
        unsafe fn id_selector_array_n(id_selector: *const i32) -> usize;
        unsafe fn id_selector_batch_new(n: i64, ids: *const i64) -> *mut i32;
        unsafe fn id_selector_bitmap_new(nbits: i64, bits: *const u8) -> *mut i32;
        unsafe fn id_selector_not_new(selector: *const i32) -> *mut i32;
        unsafe fn id_selector_all_new() -> *mut i32;
        unsafe fn id_selector_and_new(selector1: *const i32, selector2: *const i32) -> *mut i32;
        unsafe fn id_selector_or_new(selector1: *const i32, selector2: *const i32) -> *mut i32;
        unsafe fn id_selector_xor_new(selector1: *const i32, selector2: *const i32) -> *mut i32;
    }
}

pub type IDSelectorPtr = *mut i32;
use tracing::*;

pub trait IDSelectorTrait {
    fn ptr(&self) -> IDSelectorPtr;
    fn is_member(&self, id: i64) -> bool {
        unsafe { ffi::id_selector_is_member(self.ptr(), id) }
    }
    fn not(self) -> IDSelectorNot
    where
        Self: Sized + 'static,
    {
        trace!("create id_selector_not with inner={:?}", self.ptr());
        IDSelectorNot {
            inner: unsafe { ffi::id_selector_not_new(self.ptr()) },
            src: Box::new(self),
        }
    }
    fn and(self, rhs: impl IDSelectorTrait + 'static) -> IDSelectorAnd
    where
        Self: Sized + 'static,
    {
        trace!(
            "create id_selector_and with lhs={:?}, rhs={:?}",
            self.ptr(),
            rhs.ptr()
        );
        IDSelectorAnd {
            inner: unsafe { ffi::id_selector_and_new(self.ptr(), rhs.ptr()) },
            lhs: Box::new(self),
            rhs: Box::new(rhs),
        }
    }
    fn or(self, rhs: impl IDSelectorTrait + 'static) -> IDSelectorOr
    where
        Self: Sized + 'static,
    {
        trace!(
            "create id_selector_or with lhs={:?}, rhs={:?}",
            self.ptr(),
            rhs.ptr()
        );
        IDSelectorOr {
            inner: unsafe { ffi::id_selector_or_new(self.ptr(), rhs.ptr()) },
            lhs: Box::new(self),
            rhs: Box::new(rhs),
        }
    }

    fn xor(self, rhs: impl IDSelectorTrait + 'static) -> IDSelectorXOr
    where
        Self: Sized + 'static,
    {
        trace!(
            "create id_selector_xor with lhs={:?}, rhs={:?}",
            self.ptr(),
            rhs.ptr()
        );
        IDSelectorXOr {
            inner: unsafe { ffi::id_selector_xor_new(self.ptr(), rhs.ptr()) },
            lhs: Box::new(self),
            rhs: Box::new(rhs),
        }
    }
}

macro_rules! impl_id_selector {
    ($cls: ty) => {
        impl IDSelectorTrait for $cls {
            fn ptr(&self) -> IDSelectorPtr {
                self.inner
            }
        }

        impl Drop for $cls {
            fn drop(&mut self) {
                if !self.inner.is_null() {
                    tracing::trace!("drop id_selector={:?}", self.inner);
                    unsafe { ffi::id_selector_free(self.inner) };
                }
            }
        }
    };
}

pub struct IDSelectorRarnge {
    inner: IDSelectorPtr,
}

impl_id_selector!(IDSelectorRarnge);

impl IDSelectorRarnge {
    pub fn new(imin: i64, imax: i64, assume_sorted: bool) -> Self {
        let r = Self {
            inner: unsafe { ffi::id_selector_range_new(imin, imax, assume_sorted) },
        };
        trace!(%imin, %imax, %assume_sorted, "create id_selector_range inner={:?}", r.inner);
        r
    }

    pub fn imin(&self) -> i64 {
        unsafe { ffi::id_selector_range_imin(self.inner) }
    }

    pub fn imax(&self) -> i64 {
        unsafe { ffi::id_selector_range_imax(self.inner) }
    }

    pub fn assume_sorted(&self) -> bool {
        unsafe { ffi::id_selector_range_assume_sorted(self.inner) }
    }
}

pub struct IDSelectorArray {
    inner: IDSelectorPtr,
}

impl_id_selector!(IDSelectorArray);

impl IDSelectorArray {
    pub fn new(ids: &[i64]) -> Self {
        let r = Self {
            inner: unsafe { ffi::id_selector_array_new(ids.len() as i64, ids.as_ptr()) },
        };
        trace!(
            "create id_selector_array inner={:?}, n={}",
            r.inner,
            ids.len()
        );
        r
    }

    pub fn ids(&self) -> &[i64] {
        unsafe {
            let ptr = ffi::id_selector_array_ids(self.inner);
            let len = ffi::id_selector_array_n(self.inner);
            std::slice::from_raw_parts(ptr, len)
        }
    }
}

pub struct IDSelectorBatch {
    inner: IDSelectorPtr,
}
impl_id_selector!(IDSelectorBatch);

impl IDSelectorBatch {
    pub fn new(ids: &[i64]) -> Self {
        let r = Self {
            inner: unsafe { ffi::id_selector_batch_new(ids.len() as i64, ids.as_ptr()) },
        };
        trace!(
            "create id_selector_batch inner={:?}, n={}",
            r.inner,
            ids.len()
        );
        r
    }
}

pub struct IDSelectorBitmap {
    inner: IDSelectorPtr,
}

impl_id_selector!(IDSelectorBitmap);

impl IDSelectorBitmap {
    pub fn new(bits: &[u8]) -> Self {
        let n = bits.len() as i64;

        let r = Self {
            inner: unsafe { ffi::id_selector_bitmap_new(n, bits.as_ptr()) },
        };
        trace!(
            "create id_selector_bitmap inner={:?}, n={}",
            r.inner,
            bits.len()
        );
        r
    }
}

pub struct IDSelectorNot {
    inner: IDSelectorPtr,
    #[allow(unused)]
    src: Box<dyn IDSelectorTrait>,
}

impl_id_selector!(IDSelectorNot);

#[derive(smart_default::SmartDefault)]
pub struct IDSelectorAll {
    #[default(unsafe { ffi::id_selector_all_new() })]
    inner: IDSelectorPtr,
}
impl_id_selector!(IDSelectorAll);
impl IDSelectorAll {
    pub fn new() -> Self {
        let r = Self::default();
        trace!("create id_selector_all inner={:?}", r.inner,);
        r
    }
}

pub struct IDSelectorAnd {
    inner: IDSelectorPtr,
    #[allow(unused)]
    lhs: Box<dyn IDSelectorTrait>,
    #[allow(unused)]
    rhs: Box<dyn IDSelectorTrait>,
}

impl_id_selector!(IDSelectorAnd);

pub struct IDSelectorOr {
    inner: IDSelectorPtr,
    #[allow(unused)]
    lhs: Box<dyn IDSelectorTrait>,
    #[allow(unused)]
    rhs: Box<dyn IDSelectorTrait>,
}

impl_id_selector!(IDSelectorOr);

pub struct IDSelectorXOr {
    inner: IDSelectorPtr,
    #[allow(unused)]
    lhs: Box<dyn IDSelectorTrait>,
    #[allow(unused)]
    rhs: Box<dyn IDSelectorTrait>,
}

impl_id_selector!(IDSelectorXOr);

#[cfg(test)]
#[test]
fn test_id_selector_ok() {
    std::env::set_var("RUST_LOG", "trace");
    let _ = tracing_subscriber::fmt::try_init();
    let sel = IDSelectorRarnge::new(1, 10, true)
        .not()
        .or(IDSelectorAll::new());
    assert!(sel.is_member(0));
}
