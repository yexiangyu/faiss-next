macro_rules! faiss_rc {
    ($b:block) => {{
        let e = unsafe { $b };
        faiss_rc!(e)
    }};

    ($e:expr) => {
        match $e {
            0 => Ok(()),
            code => Err(crate::error::Error::from_code(code)),
        }
    };
}

macro_rules! define_index_impl {
    ($(#[$meta:meta])*$klass: ident, $free: ident) => (

        $(#[$meta])*
        pub struct $klass {
            inner: *mut faiss_next_sys::FaissIndex,
        }

        impl Drop for $klass {
            fn drop(&mut self) {
                if !self.inner.is_null()
                {
                    unsafe { faiss_next_sys::$free(self.inner) }
                }
            }
        }

        impl IndexPtr for $klass {
            fn ptr(&self) -> *const sys::FaissIndex {
                self.inner
            }

            fn mut_ptr(&mut self) -> *mut sys::FaissIndex {
                self.inner
            }

            fn into_ptr(mut self) ->  *mut sys::FaissIndex
            {
                let inner = self.inner;
                self.inner = std::ptr::null_mut();
                inner
            }
        }

        impl Index for $klass {}
        unsafe impl Send for $klass {}
        unsafe impl Sync for $klass {}
    );
}

macro_rules! define_id_selector_impl {
    ($(#[$meta:meta])*$klass: ident, $free: ident) => (

        $(#[$meta])*
        pub struct $klass {
            inner: *mut faiss_next_sys::$klass,
        }

        impl Drop for $klass {
            fn drop(&mut self) {
                if !self.inner.is_null()
                {
                    unsafe { faiss_next_sys::$free(self.inner as *mut _) }
                }
            }
        }

        impl IDSelector for $klass {
            fn ptr(&self) -> *const sys::FaissIDSelector {
                self.inner as *const _
            }

            fn mut_ptr(&mut self) -> *mut sys::FaissIDSelector {
                self.inner as *mut _
            }
        }

        unsafe impl Send for $klass {}
        unsafe impl Sync for $klass {}
    );
}

pub(crate) use define_id_selector_impl;
pub(crate) use define_index_impl;
pub(crate) use faiss_rc;
