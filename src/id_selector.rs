use std::ptr;

use faiss_next_sys::{
    self, FaissIDSelector, FaissIDSelectorAnd, FaissIDSelectorBatch, FaissIDSelectorNot,
    FaissIDSelectorOr, FaissIDSelectorRange, FaissIDSelectorXOr,
};

use crate::error::{check_return_code, Result};
use crate::idx::Idx;

pub trait IDSelector {
    fn as_ptr(&self) -> *const FaissIDSelector;
}

pub struct IDSelectorRange {
    ptr: *mut FaissIDSelectorRange,
}

impl IDSelectorRange {
    pub fn new(imin: i64, imax: i64) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissIDSelectorRange = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IDSelectorRange_new(
                &mut ptr, imin, imax,
            ))?;
            Ok(Self { ptr })
        }
    }

    pub fn imin(&self) -> i64 {
        unsafe { faiss_next_sys::faiss_IDSelectorRange_imin(self.ptr) }
    }

    pub fn imax(&self) -> i64 {
        unsafe { faiss_next_sys::faiss_IDSelectorRange_imax(self.ptr) }
    }
}

impl IDSelector for IDSelectorRange {
    fn as_ptr(&self) -> *const FaissIDSelector {
        self.ptr as *const FaissIDSelector
    }
}

impl Drop for IDSelectorRange {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_IDSelectorRange_free(self.ptr);
            }
        }
    }
}

pub struct IDSelectorBatch {
    ptr: *mut FaissIDSelectorBatch,
}

impl IDSelectorBatch {
    pub fn new(indices: &[i64]) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissIDSelectorBatch = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IDSelectorBatch_new(
                &mut ptr,
                indices.len(),
                indices.as_ptr(),
            ))?;
            Ok(Self { ptr })
        }
    }

    pub fn from_idx(indices: &[Idx]) -> Result<Self> {
        let ids: Vec<i64> = indices.iter().map(|&id| id.as_repr()).collect();
        Self::new(&ids)
    }
}

impl IDSelector for IDSelectorBatch {
    fn as_ptr(&self) -> *const FaissIDSelector {
        self.ptr as *const FaissIDSelector
    }
}

impl Drop for IDSelectorBatch {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_IDSelector_free(self.ptr as *mut FaissIDSelector);
            }
        }
    }
}

pub struct IDSelectorNot {
    ptr: *mut FaissIDSelectorNot,
}

impl IDSelectorNot {
    pub fn new<S: IDSelector>(sel: &S) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissIDSelectorNot = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IDSelectorNot_new(
                &mut ptr,
                sel.as_ptr(),
            ))?;
            Ok(Self { ptr })
        }
    }
}

impl IDSelector for IDSelectorNot {
    fn as_ptr(&self) -> *const FaissIDSelector {
        self.ptr as *const FaissIDSelector
    }
}

impl Drop for IDSelectorNot {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_IDSelector_free(self.ptr as *mut FaissIDSelector);
            }
        }
    }
}

pub struct IDSelectorAnd {
    ptr: *mut FaissIDSelectorAnd,
}

impl IDSelectorAnd {
    pub fn new<L: IDSelector, R: IDSelector>(lhs: &L, rhs: &R) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissIDSelectorAnd = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IDSelectorAnd_new(
                &mut ptr,
                lhs.as_ptr(),
                rhs.as_ptr(),
            ))?;
            Ok(Self { ptr })
        }
    }
}

impl IDSelector for IDSelectorAnd {
    fn as_ptr(&self) -> *const FaissIDSelector {
        self.ptr as *const FaissIDSelector
    }
}

impl Drop for IDSelectorAnd {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_IDSelector_free(self.ptr as *mut FaissIDSelector);
            }
        }
    }
}

pub struct IDSelectorOr {
    ptr: *mut FaissIDSelectorOr,
}

impl IDSelectorOr {
    pub fn new<L: IDSelector, R: IDSelector>(lhs: &L, rhs: &R) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissIDSelectorOr = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IDSelectorOr_new(
                &mut ptr,
                lhs.as_ptr(),
                rhs.as_ptr(),
            ))?;
            Ok(Self { ptr })
        }
    }
}

impl IDSelector for IDSelectorOr {
    fn as_ptr(&self) -> *const FaissIDSelector {
        self.ptr as *const FaissIDSelector
    }
}

impl Drop for IDSelectorOr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_IDSelector_free(self.ptr as *mut FaissIDSelector);
            }
        }
    }
}

pub struct IDSelectorXOr {
    ptr: *mut FaissIDSelectorXOr,
}

impl IDSelectorXOr {
    pub fn new<L: IDSelector, R: IDSelector>(lhs: &L, rhs: &R) -> Result<Self> {
        unsafe {
            let mut ptr: *mut FaissIDSelectorXOr = ptr::null_mut();
            check_return_code(faiss_next_sys::faiss_IDSelectorXOr_new(
                &mut ptr,
                lhs.as_ptr(),
                rhs.as_ptr(),
            ))?;
            Ok(Self { ptr })
        }
    }
}

impl IDSelector for IDSelectorXOr {
    fn as_ptr(&self) -> *const FaissIDSelector {
        self.ptr as *const FaissIDSelector
    }
}

impl Drop for IDSelectorXOr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                faiss_next_sys::faiss_IDSelector_free(self.ptr as *mut FaissIDSelector);
            }
        }
    }
}
