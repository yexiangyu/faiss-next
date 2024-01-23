use bindgen::Builder;

fn main() {
    println!("cargo:rerun-if-changed=faiss.h");
    println!("cargo:rustc-link-lib=faiss_c");
    println!("cargo:rustc-link-lib=faiss");

    let mut builder = Builder::default()
        .header("faiss.h")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .layout_tests(false)
        .allowlist_function("faiss_.*")
        .allowlist_type("idx_t|Faiss.*")
        .opaque_type("FILE")
        .wrap_unsafe_ops(true);

    #[cfg(feature = "gpu")]
    {
        builder = builder.clang_args(&[
            "-DFAISS_USE_GPU",
            "-I/usr/local/cuda/targets/x86_64-linux/include",
        ]);
    }

    if let Some(dir) = get_faiss_inc_dir() {
        builder = builder.clang_arg(format!("-I{}", dir));
    }

    if let Some(dir) = get_faiss_lib_dir() {
        println!("cargo:rustc-link-search=native={}", dir);
    }
    let bindings = builder.generate().expect("failed to build bindings");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("could not write bindings");
}

fn get_faiss_lib_dir() -> Option<String> {
    if let Ok(dir) = std::env::var("FAISS_LIB_DIR") {
        return Some(dir);
    }

    if let Ok(dir) = std::env::var("FAISS_DIR") {
        return Some(format!("{}{}lib", dir, std::path::MAIN_SEPARATOR));
    }

    None
}

fn get_faiss_inc_dir() -> Option<String> {
    if let Ok(dir) = std::env::var("FAISS_INC_DIR") {
        return Some(dir);
    }

    if let Ok(dir) = std::env::var("FAISS_DIR") {
        return Some(format!("{}{}include", dir, std::path::MAIN_SEPARATOR));
    }

    None
}
