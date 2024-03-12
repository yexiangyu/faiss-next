use crate::{
    index_binary::{impl_index_binary_drop, impl_index_binary_trait, IndexBinaryPtr},
    metric::MetricType,
};
use tracing::*;

#[cxx::bridge]
#[allow(clippy::missing_safety_doc)]
pub mod ffi {
    unsafe extern "C++" {
        include!("faiss-next/src/cpp/index_factory.hpp");
        unsafe fn index_factory(d: i32, description: *const c_char, metric: i32) -> *mut i32;
        unsafe fn index_binary_factory(d: i32, description: *const c_char) -> *mut i32;
    }
}

use crate::index::{impl_index_drop, impl_index_trait, IndexPtr};

#[derive(Debug)]
pub struct IndexImpl {
    inner: IndexPtr,
}
impl_index_drop!(IndexImpl);
impl_index_trait!(IndexImpl);

pub fn index_factory(d: usize, description: impl AsRef<str>, metric: MetricType) -> IndexImpl {
    let description = std::ffi::CString::new(description.as_ref()).expect("?");
    let inner = unsafe { ffi::index_factory(d as i32, description.as_ptr(), metric as i32) };
    trace!(
        "create index by factory with d={}, description={:?}, metric={:?}",
        d,
        description,
        metric
    );
    IndexImpl { inner }
}

#[derive(Debug)]
pub struct IndexBinaryImpl {
    inner: IndexBinaryPtr,
}
impl_index_binary_drop!(IndexBinaryImpl);
impl_index_binary_trait!(IndexBinaryImpl);

pub fn index_binary_factory(d: i32, description: impl AsRef<str>) -> IndexBinaryImpl {
    let description = std::ffi::CString::new(description.as_ref()).expect("?");
    let inner = unsafe { ffi::index_binary_factory(d, description.as_ptr()) };
    trace!(
        "create index_binary by factory with d={}, description={:?}",
        d,
        description
    );
    IndexBinaryImpl { inner }
}

#[cfg(test)]
#[test]
fn test_index_factory_ok() {
    std::env::set_var("RUST_LOG", "trace");
    use tracing::*;
    let _ = tracing_subscriber::fmt::try_init();
    let version = crate::index::version();
    use crate::index::IndexTrait;
    info!(?version);
    let description = "Flat";
    let index = index_factory(10, description, MetricType::L2);
    let d = index.d();
    let verbose = index.verbose();
    let is_trained = index.is_trained();
    let metric = index.metric_type().expect("?");
    let metric_arg = index.metric_arg();
    info!(?index, ?d, ?verbose, ?is_trained, ?metric, ?metric_arg);
}
