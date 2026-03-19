macro_rules! impl_faiss_drop {
    ($type:ident, $free_fn:ident) => {
        impl Drop for $type {
            fn drop(&mut self) {
                if !self.inner.is_null() {
                    tracing::trace!(name = stringify!($type), "dropping");
                    unsafe { crate::bindings::$free_fn(self.inner) }
                }
            }
        }
    };
}

macro_rules! impl_index_common {
    ($type:ident) => {
        impl $type {
            pub fn d(&self) -> i32 {
                unsafe { crate::bindings::faiss_Index_d(self.inner) }
            }

            pub fn ntotal(&self) -> i64 {
                unsafe { crate::bindings::faiss_Index_ntotal(self.inner) }
            }

            pub fn is_trained(&self) -> bool {
                unsafe { crate::bindings::faiss_Index_is_trained(self.inner) != 0 }
            }

            pub fn metric_type(&self) -> crate::bindings::FaissMetricType {
                unsafe { crate::bindings::faiss_Index_metric_type(self.inner) }
            }

            pub fn verbose(&self) -> bool {
                unsafe { crate::bindings::faiss_Index_verbose(self.inner) != 0 }
            }

            pub fn set_verbose(&mut self, verbose: bool) {
                unsafe { crate::bindings::faiss_Index_set_verbose(self.inner, verbose as i32) }
            }

            pub fn train(&mut self, n: i64, x: &[f32]) -> crate::error::Result<()> {
                crate::error::check_return_code(unsafe {
                    crate::bindings::faiss_Index_train(self.inner, n, x.as_ptr())
                })
            }

            pub fn add(&mut self, n: i64, x: &[f32]) -> crate::error::Result<()> {
                crate::error::check_return_code(unsafe {
                    crate::bindings::faiss_Index_add(self.inner, n, x.as_ptr())
                })
            }

            pub fn add_with_ids(
                &mut self,
                n: i64,
                x: &[f32],
                ids: &[i64],
            ) -> crate::error::Result<()> {
                crate::error::check_return_code(unsafe {
                    crate::bindings::faiss_Index_add_with_ids(
                        self.inner,
                        n,
                        x.as_ptr(),
                        ids.as_ptr(),
                    )
                })
            }

            pub fn search(
                &self,
                n: i64,
                x: &[f32],
                k: i64,
                distances: &mut [f32],
                labels: &mut [i64],
            ) -> crate::error::Result<()> {
                crate::error::check_return_code(unsafe {
                    crate::bindings::faiss_Index_search(
                        self.inner,
                        n,
                        x.as_ptr(),
                        k,
                        distances.as_mut_ptr(),
                        labels.as_mut_ptr(),
                    )
                })
            }

            pub fn range_search(
                &self,
                n: i64,
                x: &[f32],
                radius: f32,
                result: &mut crate::bindings::FaissRangeSearchResult,
            ) -> crate::error::Result<()> {
                crate::error::check_return_code(unsafe {
                    crate::bindings::faiss_Index_range_search(
                        self.inner,
                        n,
                        x.as_ptr(),
                        radius,
                        result,
                    )
                })
            }

            pub fn reset(&mut self) -> crate::error::Result<()> {
                crate::error::check_return_code(unsafe {
                    crate::bindings::faiss_Index_reset(self.inner)
                })
            }

            pub fn reconstruct(&self, key: i64, recons: &mut [f32]) -> crate::error::Result<()> {
                crate::error::check_return_code(unsafe {
                    crate::bindings::faiss_Index_reconstruct(self.inner, key, recons.as_mut_ptr())
                })
            }
        }
    };
}

macro_rules! impl_index_binary_common {
    ($type:ident) => {
        impl $type {
            pub fn d(&self) -> i32 {
                unsafe { crate::bindings::faiss_IndexBinary_d(self.inner) }
            }

            pub fn ntotal(&self) -> i64 {
                unsafe { crate::bindings::faiss_IndexBinary_ntotal(self.inner) }
            }

            pub fn is_trained(&self) -> bool {
                unsafe { crate::bindings::faiss_IndexBinary_is_trained(self.inner) != 0 }
            }

            pub fn metric_type(&self) -> crate::bindings::FaissMetricType {
                unsafe { crate::bindings::faiss_IndexBinary_metric_type(self.inner) }
            }

            pub fn train(&mut self, n: i64, x: &[u8]) -> crate::error::Result<()> {
                crate::error::check_return_code(unsafe {
                    crate::bindings::faiss_IndexBinary_train(self.inner, n, x.as_ptr())
                })
            }

            pub fn add(&mut self, n: i64, x: &[u8]) -> crate::error::Result<()> {
                crate::error::check_return_code(unsafe {
                    crate::bindings::faiss_IndexBinary_add(self.inner, n, x.as_ptr())
                })
            }

            pub fn add_with_ids(
                &mut self,
                n: i64,
                x: &[u8],
                ids: &[i64],
            ) -> crate::error::Result<()> {
                crate::error::check_return_code(unsafe {
                    crate::bindings::faiss_IndexBinary_add_with_ids(
                        self.inner,
                        n,
                        x.as_ptr(),
                        ids.as_ptr(),
                    )
                })
            }

            pub fn search(
                &self,
                n: i64,
                x: &[u8],
                k: i64,
                distances: &mut [i32],
                labels: &mut [i64],
            ) -> crate::error::Result<()> {
                crate::error::check_return_code(unsafe {
                    crate::bindings::faiss_IndexBinary_search(
                        self.inner,
                        n,
                        x.as_ptr(),
                        k,
                        distances.as_mut_ptr(),
                        labels.as_mut_ptr(),
                    )
                })
            }

            pub fn reset(&mut self) -> crate::error::Result<()> {
                crate::error::check_return_code(unsafe {
                    crate::bindings::faiss_IndexBinary_reset(self.inner)
                })
            }
        }
    };
}

pub(crate) use impl_faiss_drop;
pub(crate) use impl_index_binary_common;
pub(crate) use impl_index_common;
