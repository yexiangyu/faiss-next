macro_rules! rc {
    ($b:block) => {{
        let e = unsafe { $b };
        rc!(e)
    }};

    ($e:expr) => {
        match $e {
            0 => Ok(()),
            code => Err(crate::error::Error::from_code(code)),
        }
    };
}

macro_rules! declare_index {
    ($(#[$meta:meta])*$klass: ident) => (

        $(#[$meta])*
        pub struct $klass {
            pub inner: *mut faiss_next_sys::$klass
        }
    );

    ($(#[$meta:meta])*$klass: ident, $inner: ty) => (

        $(#[$meta])*
        pub struct $klass {
            pub inner: *mut $inner
        }
    );
}

macro_rules! impl_index_drop {
    ($klass: ident, $drop: ident) => {
        impl Drop for $klass {
            fn drop(&mut self) {
                unsafe { faiss_next_sys::$drop(self.inner) }
            }
        }
    };
}

macro_rules! impl_index_owned_ptr {
    ($klass: ident) => {
        impl FaissIndexOwnedPtr for $klass {
            fn into_ptr(self) -> *mut sys::FaissIndex {
                let inner = self.inner;
                std::mem::forget(self);
                inner
            }
        }
    };
}

macro_rules! impl_index_borrowed_ptr {
    ($klass: ident) => {
        impl FaissIndexBorrowedPtr for $klass {
            fn to_ptr(&self) -> *const sys::FaissIndex {
                self.inner
            }
        }
    };
}

macro_rules! impl_index_mut_ptr {
    ($klass: ident) => {
        impl FaissIndexMutPtr for $klass {
            fn to_mut(&mut self) -> *mut sys::FaissIndex {
                self.inner
            }
        }
    };
    () => {};
}

pub(crate) use declare_index;
pub(crate) use impl_index_borrowed_ptr;
pub(crate) use impl_index_drop;
pub(crate) use impl_index_mut_ptr;
pub(crate) use impl_index_owned_ptr;
pub(crate) use rc;

// macro_rules! define_index_impl {
//     ($(#[$meta:meta])*$klass: ident, $free: ident) => (

//         $(#[$meta])*
//         pub struct $klass {
//             pub inner: *mut faiss_next_sys::FaissIndex,
//         }

//         impl Drop for $klass {
//             fn drop(&mut self) {
//                 unsafe { faiss_next_sys::$free(self.inner) }
//             }
//         }

//         impl IndexPtr for $klass {
//             fn ptr(&self) -> *const sys::FaissIndex {
//                 self.inner
//             }

//             fn mut_ptr(&mut self) -> *mut sys::FaissIndex {
//                 self.inner
//             }

//             fn into_ptr(self) ->  *mut sys::FaissIndex
//             {
//                 let inner = self.inner;
//                 std::mem::forget(self);
//                 inner
//             }
//         }

//         impl Index for $klass {}
//         unsafe impl Send for $klass {}
//         unsafe impl Sync for $klass {}
//     );
// }

// macro_rules! define_id_selector_impl {
//     ($(#[$meta:meta])*$klass: ident, $free: ident) => (

//         $(#[$meta])*
//         pub struct $klass {
//             inner: *mut faiss_next_sys::$klass,
//         }

//         impl Drop for $klass {
//             fn drop(&mut self) {
//                 if !self.inner.is_null()
//                 {
//                     unsafe { faiss_next_sys::$free(self.inner as *mut _) }
//                 }
//             }
//         }

//         impl IDSelector for $klass {
//             fn ptr(&self) -> *const sys::FaissIDSelector {
//                 self.inner as *const _
//             }

//             fn mut_ptr(&mut self) -> *mut sys::FaissIDSelector {
//                 self.inner as *mut _
//             }
//         }

//         unsafe impl Send for $klass {}
//         unsafe impl Sync for $klass {}
//     );
// }

// macro_rules! define_vector_transform_impl {
//     ($(#[$meta:meta])*$klass: ident, $free: ident) => (

//         $(#[$meta])*
//         pub struct $klass {
//             inner: *mut faiss_next_sys::$klass,
//         }

//         impl Drop for $klass {
//             fn drop(&mut self) {
//                 unsafe { faiss_next_sys::$free(self.inner) }
//             }
//         }

//         impl VectorTransformPtr for $klass {
//             fn ptr(&self) -> *const sys::FaissVectorTransform {
//                 self.inner as *const _
//             }

//             fn mut_ptr(&mut self) -> *mut sys::FaissVectorTransform {
//                 self.inner
//             }

//             fn into_ptr(self) -> *mut sys::FaissVectorTransform {
//                 let inner = self.inner;
//                 std::mem::forget(self);
//                 inner
//             }
//         }

//         impl VectorTransform for $klass {}
//         impl LinearTransform for $klass {}
//         unsafe impl Send for $klass {}
//         unsafe impl Sync for $klass {}
//     );
// }

// pub(crate) use define_id_selector_impl;
// pub(crate) use define_index_impl;
// pub(crate) use define_vector_transform_impl;
