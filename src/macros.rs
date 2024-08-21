#![allow(unused_macros)]

macro_rules! impl_faiss_drop {
    ($klass: ident, $drop_fun: ident,  $inner: ty) => {
        impl Drop for $klass {
            fn drop(&mut self) {
                unsafe { ffi::$drop_fun(self.inner) }
            }
        }
    };
}

macro_rules! impl_faiss_setter {
    ($klass: ident, $setter: ident, $inner_setter: ident, $val: ident, $val_ty: ty) => {
        impl $klass {
            pub fn $setter(&mut self, $val: $val_ty) {
                unsafe { faiss_next_sys::$inner_setter(self.inner, $val) }
            }
        }
    };
}

macro_rules! impl_faiss_getter {
    ($klass: ident, $getter: ident, $inner_getter: ident, $ret: ty) => {
        impl $klass {
            pub fn $getter(&self) -> $ret {
                unsafe { faiss_next_sys::$inner_getter(self.inner) }
            }
        }
    };
}

macro_rules! impl_faiss_new {
    ($outer :ident, $outer_new: ident, $inner: ident, $inner_new: ident) => {
        impl $outer {
            pub fn $outer_new() -> crate::error::Result<Self> {
                let mut inner = null_mut();
                crate::error::faiss_rc(unsafe { faiss_next_sys::$inner_new(&mut inner) })?;
                Self { inner }
            }
        }
    };

    ($outer :ident, $outer_new: ident, $inner: ident, $inner_new: ident, $arg1: ident, $arg1_ty: ty) => {
        impl $outer {
            #[allow(clippy::not_unsafe_ptr_arg_deref)]
            pub fn $outer_new($arg1: $arg1_ty) -> crate::error::Result<Self> {
                let mut inner = std::ptr::null_mut();
                crate::error::faiss_rc(unsafe { faiss_next_sys::$inner_new(&mut inner, $arg1) })?;
                Ok(Self { inner })
            }
        }
    };

    ($outer :ident, $outer_new: ident, $inner: ident, $inner_new: ident, $arg1: ident, $arg1_ty: ty, $arg2: ident, $arg2_ty: ty) => {
        impl $outer {
            pub fn $outer_new($arg1: $arg1_ty, $arg2: $arg2_ty) -> crate::error::Result<Self> {
                let mut inner = std::ptr::null_mut();
                crate::error::faiss_rc(unsafe {
                    faiss_next_sys::$inner_new(&mut inner, $arg1, $arg2)
                })?;
                Ok(Self { inner })
            }
        }
    };

    ($outer :ident, $outer_new: ident, $inner: ident, $inner_new: ident, $arg1: ident, $arg1_ty: ty, $arg2: ident, $arg2_ty: ty, $arg3: ident, $arg3_ty: ty) => {
        impl $outer {
            #[allow(clippy::not_unsafe_ptr_arg_deref)]
            pub fn $outer_new(
                $arg1: $arg1_ty,
                $arg2: $arg2_ty,
                $arg3: $arg3_ty,
            ) -> crate::error::Result<Self> {
                let mut inner = std::ptr::null_mut();
                crate::error::faiss_rc(unsafe {
                    faiss_next_sys::$inner_new(&mut inner, $arg1, $arg2, $arg3)
                })?;
                Ok(Self { inner })
            }
        }
    };

    ($outer :ident, $outer_new: ident, $inner: ident, $inner_new: ident, $arg1: ident, $arg1_ty: ty, $arg2: ident, $arg2_ty: ty, $arg3: ident, $arg3_ty: ty, $arg4: ident, $arg4_ty: ty) => {
        impl $outer {
            pub fn $outer_new(
                $arg1: $arg1_ty,
                $arg2: $arg2_ty,
                $arg3: $arg3_ty,
                $arg4: $arg4_ty,
            ) -> crate::error::Result<Self> {
                let mut inner = std::ptr::null_mut();
                crate::error::faiss_rc(unsafe {
                    faiss_next_sys::$inner_new(&mut inner, $arg1, $arg2, $arg3, $arg4)
                })?;
                Ok(Self { inner })
            }
        }
    };
}

macro_rules! impl_faiss_drop {
    ($klass: ident, $drop_fun: ident) => {
        impl Drop for $klass {
            fn drop(&mut self) {
                unsafe { faiss_next_sys::$drop_fun(self.inner) }
            }
        }
    };
}

macro_rules! impl_faiss_drop_as {
    ($klass: ident, $drop_fun: ident) => {
        impl Drop for $klass {
            fn drop(&mut self) {
                unsafe { faiss_next_sys::$drop_fun(self.inner as *mut _) }
            }
        }
    };
}

macro_rules! impl_faiss_functioin_rc {
    ($klass: ident, $fun: ident, $inner_fun: ident) => {
        impl $klass {
            pub fn $fun(&self) -> crate::error::Result<()> {
                crate::error::faiss_rc(unsafe { faiss_next_sys::$inner_fun(self.inner) })
            }
        }
    };

    ($klass: ident, $fun: ident, $inner_fun: ident, $arg1: ident, $arg1_ty: ty) => {
        impl $klass {
            pub fn $fun(&self, $arg1: $arg1_ty) -> crate::error::Result<()> {
                crate::error::faiss_rc(unsafe { faiss_next_sys::$inner_fun(self.inner, $arg1) })
            }
        }
    };

    ($klass: ident, $fun: ident, $inner_fun: ident, $arg1: ident, $arg1_ty: ty, $arg2: ident, $arg2_ty: ty) => {
        impl $klass {
            pub fn $fun(&self, $arg1: $arg1_ty, $arg2: $arg2_ty) -> crate::error::Result<()> {
                crate::error::faiss_rc(unsafe {
                    faiss_next_sys::$inner_fun(self.inner, $arg1, $arg2)
                })
            }
        }
    };

    ($klass: ident, $fun: ident, $inner_fun: ident, $arg1: ident, $arg1_ty: ty, $arg2: ident, $arg2_ty: ty, $arg3: ident, $arg3_ty: ty) => {
        impl $klass {
            pub fn $fun(
                &self,
                $arg1: $arg1_ty,
                $arg2: $arg2_ty,
                $arg3: $arg3_ty,
            ) -> crate::error::Result<()> {
                crate::error::faiss_rc(unsafe {
                    faiss_next_sys::$inner_fun(self.inner, $arg1, $arg2, $arg3)
                })
            }
        }
    };

    ($klass: ident, $fun: ident, $inner_fun: ident, $arg1: ident, $arg1_ty: ty, $arg2: ident, $arg2_ty: ty, $arg3: ident, $arg3_ty: ty, $arg4: ident, $arg4_ty: ty) => {
        impl $klass {
            #[allow(clippy::not_unsafe_ptr_arg_deref)]
            pub fn $fun(
                &self,
                $arg1: $arg1_ty,
                $arg2: $arg2_ty,
                $arg3: $arg3_ty,
                $arg4: $arg4_ty,
            ) -> crate::error::Result<()> {
                crate::error::faiss_rc(unsafe {
                    faiss_next_sys::$inner_fun(self.inner, $arg1, $arg2, $arg3, $arg4)
                })
            }
        }
    };
}

macro_rules! impl_faiss_functioin_static_rc {
    ($klass: ident, $fun: ident, $inner_fun: ident) => {
        impl $klass {
            fn $fun() -> crate::error::Result<()> {
                crate::error::faiss_rc(unsafe { faiss_next_sys::$inner_fun(self.inner) })
            }
        }
    };

    ($klass: ident, $fun: ident, $inner_fun: ident, $arg1: ident, $arg1_ty: ty) => {
        impl $klass {
            fn $fun($arg1: $arg1_ty) -> crate::error::Result<()> {
                crate::error::faiss_rc(unsafe { faiss_next_sys::$inner_fun(self.inner, $arg1) })
            }
        }
    };

    ($klass: ident, $fun: ident, $inner_fun: ident, $arg1: ident, $arg1_ty: ty, $arg2: ident, $arg2_ty: ty) => {
        impl $klass {
            fn $fun($arg1: $arg1_ty, $arg2: $arg2_ty) -> crate::error::Result<()> {
                crate::error::faiss_rc(unsafe { faiss_next_sys::$inner_fun($arg1, $arg2) })
            }
        }
    };

    ($klass: ident, $fun: ident, $inner_fun: ident, $arg1: ident, $arg1_ty: ty, $arg2: ident, $arg2_ty: ty, $arg3: ident, $arg3_ty: ty) => {
        impl $klass {
            fn $fun($arg1: $arg1_ty, $arg2: $arg2_ty, $arg3: $arg3_ty) -> crate::error::Result<()> {
                crate::error::faiss_rc(unsafe { faiss_next_sys::$inner_fun($arg1, $arg2, $arg3) })
            }
        }
    };
}

macro_rules! impl_faiss_functioin_void {
    ($klass: ident, $fun: ident, $inner_fun: ident) => {
        impl $klass {
            pub fn $fun(&self) {
                unsafe { faiss_next_sys::$inner_fun(self.inner) }
            }
        }
    };

    ($klass: ident, $fun: ident, $inner_fun: ident, $arg1: ident, $arg1_ty: ty) => {
        impl $klass {
            fn $fun(&self, $arg1: $arg1_ty) {
                unsafe { faiss_next_sys::$inner_fun(self.inner, $arg1) }
            }
        }
    };

    ($klass: ident, $fun: ident, $inner_fun: ident, $arg1: ident, $arg1_ty: ty, $arg2: ident, $arg2_ty: ty) => {
        impl $klass {
            fn $fun(&self, $arg1: $arg1_ty, $arg2: $arg2_ty) {
                unsafe { faiss_next_sys::$inner_fun(self.inner, $arg1, $arg2) }
            }
        }
    };
}

pub(crate) use impl_faiss_drop;
pub(crate) use impl_faiss_drop_as;
pub(crate) use impl_faiss_functioin_rc;
pub(crate) use impl_faiss_functioin_static_rc;
pub(crate) use impl_faiss_functioin_void;
pub(crate) use impl_faiss_getter;
pub(crate) use impl_faiss_new;
pub(crate) use impl_faiss_setter;
