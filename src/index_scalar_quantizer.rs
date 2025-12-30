use crate::error::Result;
use faiss_next_sys as ffi;

use std::ptr::null_mut;

use crate::index::{IndexTrait, MetricType};

pub use ffi::FaissQuantizerType;

pub trait IndexScalarQuantizerTrait: IndexTrait {}

macro_rules! impl_index_scalar_quantizer {
    ($cls: ident) => {
        impl IndexTrait for $cls {
            fn inner(&self) -> *mut ffi::FaissIndex {
                self.inner
            }
        }

        impl IndexScalarQuantizerTrait for $cls {}
    };
}

#[derive(Debug)]
pub struct IndexScalarQuantizer {
    inner: *mut ffi::FaissIndex,
}

impl_index_scalar_quantizer!(IndexScalarQuantizer);
ffi::impl_drop!(IndexScalarQuantizer, faiss_IndexScalarQuantizer_free);

impl IndexScalarQuantizer {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexScalarQuantizer_new, &mut inner)?;
        Ok(Self { inner })
    }

    pub fn new_with(d: i64, qt: ffi::FaissQuantizerType, metric: MetricType) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(
            faiss_IndexScalarQuantizer_new_with,
            &mut inner,
            d,
            qt,
            metric
        )?;
        Ok(Self { inner })
    }

    pub fn cast(index: impl IndexTrait) -> Result<Self> {
        let inner = ffi::run!(faiss_IndexScalarQuantizer_cast, index.inner());
        if inner.is_null() {
            return Err(crate::error::Error::Faiss(ffi::Error {
                code: -1,
                message: "Failed to cast index".to_string(),
            }));
        }
        Ok(Self { inner })
    }
}

pub trait IndexIVFScalarQuantizerTrait: IndexTrait {}

macro_rules! impl_index_ivf_scalar_quantizer {
    ($cls: ident) => {
        impl IndexTrait for $cls {
            fn inner(&self) -> *mut ffi::FaissIndex {
                self.inner
            }
        }

        impl IndexIVFScalarQuantizerTrait for $cls {}
    };
}

#[derive(Debug)]
pub struct IndexIVFScalarQuantizer {
    inner: *mut ffi::FaissIndex,
}

impl_index_ivf_scalar_quantizer!(IndexIVFScalarQuantizer);
ffi::impl_drop!(IndexIVFScalarQuantizer, faiss_IndexIVFScalarQuantizer_free);
#[rustfmt::skip]
ffi::impl_getter!(IndexIVFScalarQuantizer, nlist, faiss_IndexIVFScalarQuantizer_nlist, usize);
#[rustfmt::skip]
ffi::impl_getter!(IndexIVFScalarQuantizer, nprobe, faiss_IndexIVFScalarQuantizer_nprobe, usize);
#[rustfmt::skip]
ffi::impl_setter!(IndexIVFScalarQuantizer, set_nprobe, faiss_IndexIVFScalarQuantizer_set_nprobe, val, usize);

impl IndexIVFScalarQuantizer {
    pub fn new() -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(faiss_IndexIVFScalarQuantizer_new, &mut inner)?;
        let mut ret = Self { inner };
        ret.set_own_fields(true)?;
        Ok(ret)
    }

    pub fn new_with(
        d: i64,
        quantizer: impl IndexTrait,
        nlist: usize,
        qt: ffi::FaissQuantizerType,
    ) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(
            faiss_IndexIVFScalarQuantizer_new_with,
            &mut inner,
            quantizer.inner(),
            d,
            nlist,
            qt
        )?;
        let mut ret = Self { inner };
        ret.set_own_fields(true)?;
        Ok(ret)
    }

    pub fn new_with_metric(
        d: usize,
        quantizer: impl IndexTrait,
        nlist: usize,
        qt: ffi::FaissQuantizerType,
        metric: MetricType,
        encode_residual: bool,
    ) -> Result<Self> {
        let mut inner = null_mut();
        ffi::ok!(
            faiss_IndexIVFScalarQuantizer_new_with_metric,
            &mut inner,
            quantizer.inner(),
            d,
            nlist,
            qt,
            metric,
            encode_residual as i32
        )?;
        let mut ret = Self { inner };
        ret.set_own_fields(true)?;
        Ok(ret)
    }

    pub fn cast(index: impl IndexTrait) -> Result<Self> {
        let inner = ffi::run!(faiss_IndexIVFScalarQuantizer_cast, index.inner());
        if inner.is_null() {
            return Err(crate::error::Error::Faiss(ffi::Error {
                code: -1,
                message: "Failed to cast index".to_string(),
            }));
        }
        Ok(Self { inner })
    }

    fn set_own_fields(&mut self, own: bool) -> Result<()> {
        unsafe { ffi::faiss_IndexIVFScalarQuantizer_set_own_fields(self.inner, own as i32) };
        Ok(())
    }

    pub fn own_fields(&self) -> bool {
        unsafe { ffi::faiss_IndexIVFScalarQuantizer_own_fields(self.inner) > 0 }
    }

    pub fn quantizer(&self) -> Option<IndexScalarQuantizer> {
        let inner = unsafe { ffi::faiss_IndexIVFScalarQuantizer_quantizer(self.inner) };
        match inner.is_null() {
            true => None,
            false => Some(IndexScalarQuantizer { inner }),
        }
    }

    pub fn add_core(
        &mut self,
        x: impl AsRef<[f32]>,
        xids: impl AsRef<[i64]>,
        precomputed_idx: impl AsRef<[i64]>,
    ) -> Result<()> {
        let n = x.as_ref().len() / self.d() as usize;
        assert_eq!(xids.as_ref().len(), n);
        assert_eq!(precomputed_idx.as_ref().len(), n);
        ffi::ok!(
            faiss_IndexIVFScalarQuantizer_add_core,
            self.inner,
            n as i64,
            x.as_ref().as_ptr(),
            xids.as_ref().as_ptr(),
            precomputed_idx.as_ref().as_ptr()
        )?;
        Ok(())
    }
}
