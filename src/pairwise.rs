use faiss_next_sys;

pub fn pairwise_l2_sqr(d: usize, x: &[f32], y: &[f32]) -> Vec<f32> {
    let nq = x.len() / d;
    let nb = y.len() / d;
    let mut dis = vec![0.0f32; nq * nb];
    unsafe {
        faiss_next_sys::faiss_pairwise_L2sqr_with_defaults(
            d as i64,
            nq as i64,
            x.as_ptr(),
            nb as i64,
            y.as_ptr(),
            dis.as_mut_ptr(),
        );
    }
    dis
}

pub fn pairwise_l2_sqr_with_stride(
    d: usize,
    x: &[f32],
    y: &[f32],
    ldq: i64,
    ldb: i64,
    ldd: i64,
) -> Vec<f32> {
    let nq = x.len() / if ldq > 0 { ldq as usize } else { d };
    let nb = y.len() / if ldb > 0 { ldb as usize } else { d };
    let mut dis = vec![0.0f32; nq * if ldd > 0 { ldd as usize } else { nb }];
    unsafe {
        faiss_next_sys::faiss_pairwise_L2sqr(
            d as i64,
            nq as i64,
            x.as_ptr(),
            nb as i64,
            y.as_ptr(),
            dis.as_mut_ptr(),
            ldq,
            ldb,
            ldd,
        );
    }
    dis
}

pub fn inner_products(d: usize, x: &[f32], y: &[f32]) -> Vec<f32> {
    let ny = y.len() / d;
    let mut ip = vec![0.0f32; ny];
    unsafe {
        faiss_next_sys::faiss_fvec_inner_products_ny(
            ip.as_mut_ptr(),
            x.as_ptr(),
            y.as_ptr(),
            d,
            ny,
        );
    }
    ip
}

pub fn l2_sqr_ny(d: usize, x: &[f32], y: &[f32]) -> Vec<f32> {
    let ny = y.len() / d;
    let mut dis = vec![0.0f32; ny];
    unsafe {
        faiss_next_sys::faiss_fvec_L2sqr_ny(dis.as_mut_ptr(), x.as_ptr(), y.as_ptr(), d, ny);
    }
    dis
}

pub fn norm_l2_sqr(x: &[f32], d: usize) -> f32 {
    unsafe { faiss_next_sys::faiss_fvec_norm_L2sqr(x.as_ptr(), d) }
}

pub fn norms_l2(x: &[f32], d: usize) -> Vec<f32> {
    let nx = x.len() / d;
    let mut norms = vec![0.0f32; nx];
    unsafe {
        faiss_next_sys::faiss_fvec_norms_L2(norms.as_mut_ptr(), x.as_ptr(), d, nx);
    }
    norms
}

pub fn norms_l2_sqr(x: &[f32], d: usize) -> Vec<f32> {
    let nx = x.len() / d;
    let mut norms = vec![0.0f32; nx];
    unsafe {
        faiss_next_sys::faiss_fvec_norms_L2sqr(norms.as_mut_ptr(), x.as_ptr(), d, nx);
    }
    norms
}

pub fn renorm_l2(x: &mut [f32], d: usize) {
    let nx = x.len() / d;
    unsafe {
        faiss_next_sys::faiss_fvec_renorm_L2(d, nx, x.as_mut_ptr());
    }
}

pub fn set_distance_compute_blas_threshold(value: i32) {
    unsafe {
        faiss_next_sys::faiss_set_distance_compute_blas_threshold(value as std::os::raw::c_int);
    }
}

pub fn get_distance_compute_blas_threshold() -> i32 {
    unsafe { faiss_next_sys::faiss_get_distance_compute_blas_threshold() }
}

pub fn set_distance_compute_blas_query_bs(value: i32) {
    unsafe {
        faiss_next_sys::faiss_set_distance_compute_blas_query_bs(value as std::os::raw::c_int);
    }
}

pub fn get_distance_compute_blas_query_bs() -> i32 {
    unsafe { faiss_next_sys::faiss_get_distance_compute_blas_query_bs() }
}

pub fn set_distance_compute_blas_database_bs(value: i32) {
    unsafe {
        faiss_next_sys::faiss_set_distance_compute_blas_database_bs(value as std::os::raw::c_int);
    }
}

pub fn get_distance_compute_blas_database_bs() -> i32 {
    unsafe { faiss_next_sys::faiss_get_distance_compute_blas_database_bs() }
}
