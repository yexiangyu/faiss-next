use crate::index::traits::*;
use crate::macros::{
    declare_index, impl_index_borrowed_ptr, impl_index_drop, impl_index_mut_ptr,
    impl_index_owned_ptr,
};
use faiss_next_sys as sys;

declare_index!(FaissIndexOwned, sys::FaissIndex);

impl_index_drop!(FaissIndexOwned, faiss_Index_free);
impl_index_owned_ptr!(FaissIndexOwned);
impl_index_mut_ptr!(FaissIndexOwned);
impl_index_borrowed_ptr!(FaissIndexOwned);

impl FaissIndexConstTrait for FaissIndexOwned {}
impl FaissIndexMutTrait for FaissIndexOwned {}
impl FaissIndexOwnedTrait for FaissIndexOwned {}
