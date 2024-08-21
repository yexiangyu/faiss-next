use crate::{
    error::*, impl_aux_index_structure::FaissRangeSearchResult, macros::impl_faiss_drop,
    traits::FaissIDSelectorTrait,
};
use faiss_next_sys as ffi;

pub trait FaissIndexBinaryTrait {
    fn inner(&self) -> *mut ffi::FaissIndexBinary;

    fn d(&self) -> i32 {
        unsafe { ffi::faiss_IndexBinary_d(self.inner()) }
    }
    fn is_trained(&self) -> bool {
        unsafe { ffi::faiss_IndexBinary_is_trained(self.inner()) > 0 }
    }
    fn ntotal(&self) -> i64 {
        unsafe { ffi::faiss_IndexBinary_ntotal(self.inner()) }
    }
    fn metric_type(&self) -> ffi::FaissMetricType {
        unsafe { ffi::faiss_IndexBinary_metric_type(self.inner()) }
    }
    fn verbose(&self) -> bool {
        unsafe { ffi::faiss_IndexBinary_verbose(self.inner()) > 0 }
    }
    fn set_verbose(&mut self, verbose: bool) {
        unsafe { ffi::faiss_IndexBinary_set_verbose(self.inner(), verbose as i32) }
    }
    fn train(&mut self, x: impl AsRef<[u8]>) -> Result<()> {
        let n = x.as_ref().len() as i64 / self.d() as i64;
        faiss_rc(unsafe { ffi::faiss_IndexBinary_train(self.inner(), n, x.as_ref().as_ptr()) })
    }
    fn add(&mut self, x: impl AsRef<[u8]>) -> Result<()> {
        assert_eq!(x.as_ref().len() as i32 % self.d(), 0);
        let n = x.as_ref().len() as i64 / self.d() as i64;
        faiss_rc(unsafe { ffi::faiss_IndexBinary_add(self.inner(), n, x.as_ref().as_ptr()) })
    }
    fn add_with_ids(&mut self, x: impl AsRef<[u8]>, xids: impl AsRef<[i64]>) -> Result<()> {
        assert_eq!(x.as_ref().len() as i32 % self.d(), 0);
        let n = x.as_ref().len() as i64 / self.d() as i64;
        assert_eq!(xids.as_ref().len() as i64, n);
        faiss_rc(unsafe {
            ffi::faiss_IndexBinary_add_with_ids(
                self.inner(),
                n,
                x.as_ref().as_ptr(),
                xids.as_ref().as_ptr(),
            )
        })
    }
    fn search(
        &self,
        x: impl AsRef<[u8]>,
        k: i64,
        mut distances: impl AsMut<[i32]>,
        mut labels: impl AsMut<[i64]>,
    ) -> Result<()> {
        assert_eq!(x.as_ref().len() as i32 % self.d(), 0);
        let n = x.as_ref().len() as i64 / self.d() as i64;
        assert_eq!(n * k, distances.as_mut().len() as i64);
        assert_eq!(n * k, labels.as_mut().len() as i64);
        faiss_rc(unsafe {
            ffi::faiss_IndexBinary_search(
                self.inner(),
                n,
                x.as_ref().as_ptr(),
                k,
                distances.as_mut().as_mut_ptr(),
                labels.as_mut().as_mut_ptr(),
            )
        })
    }
    fn range_search(
        &self,
        x: impl AsRef<[u8]>,
        radius: i32,
        result: &mut FaissRangeSearchResult,
    ) -> Result<()> {
        assert_eq!(x.as_ref().len() as i32 % self.d(), 0);
        let n = x.as_ref().len() as i64 / self.d() as i64;
        faiss_rc(unsafe {
            ffi::faiss_IndexBinary_range_search(
                self.inner(),
                n,
                x.as_ref().as_ptr(),
                radius,
                result.inner,
            )
        })
    }
    fn assign(&self, x: impl AsRef<[u8]>, mut labels: impl AsMut<[i64]>, k: i64) -> Result<()> {
        assert_eq!(x.as_ref().len() as i32 % self.d(), 0);
        let n = x.as_ref().len() as i64 / self.d() as i64;
        assert_eq!(n, labels.as_mut().len() as i64);
        faiss_rc(unsafe {
            ffi::faiss_IndexBinary_assign(
                self.inner(),
                n,
                x.as_ref().as_ptr(),
                labels.as_mut().as_mut_ptr(),
                k,
            )
        })
    }
    fn reset(&mut self) -> Result<()> {
        faiss_rc(unsafe { ffi::faiss_IndexBinary_reset(self.inner()) })
    }
    fn remove_ids(&mut self, sel: impl FaissIDSelectorTrait) -> Result<usize> {
        let mut nremove = 0;
        faiss_rc(unsafe {
            ffi::faiss_IndexBinary_remove_ids(self.inner(), sel.inner(), &mut nremove)
        })?;
        Ok(nremove)
    }
    fn reconstruct(&mut self, key: i64, mut recons: impl AsMut<[u8]>) -> Result<()> {
        assert_eq!(recons.as_mut().len() as i32 % self.d(), 0);
        faiss_rc(unsafe {
            ffi::faiss_IndexBinary_reconstruct(self.inner(), key, recons.as_mut().as_mut_ptr())
        })
    }

    fn reconstruct_n(&mut self, i0: i64, mut recons: impl AsMut<[u8]>) -> Result<()> {
        assert_eq!(recons.as_mut().len() as i32 % self.d(), 0);
        let ni = recons.as_mut().len() as i64 / self.d() as i64;
        faiss_rc(unsafe {
            ffi::faiss_IndexBinary_reconstruct_n(self.inner(), i0, ni, recons.as_mut().as_mut_ptr())
        })
    }
    fn save(&self, filename: impl AsRef<str>) -> Result<()> {
        let filename = std::ffi::CString::new(filename.as_ref())?;
        faiss_rc(unsafe { ffi::faiss_write_index_binary_fname(self.inner(), filename.as_ptr()) })
    }
}

#[derive(Debug)]
pub struct FaissIndexBinaryOwned {
    pub inner: *mut ffi::FaissIndexBinary,
}
impl_faiss_drop!(FaissIndexBinaryOwned, faiss_IndexBinary_free);
impl FaissIndexBinaryTrait for FaissIndexBinaryOwned {
    fn inner(&self) -> *mut ffi::FaissIndexBinary {
        self.inner
    }
}
impl FaissIndexBinaryOwned {
    pub fn read(filename: impl AsRef<str>, io_flags: i32) -> Result<Self> {
        let filename = filename.as_ref();
        let filename = std::ffi::CString::new(filename)?;
        let mut inner = std::ptr::null_mut();
        faiss_rc(unsafe {
            ffi::faiss_read_index_binary_fname(filename.as_ptr(), io_flags, &mut inner)
        })?;
        Ok(Self { inner })
    }
}
