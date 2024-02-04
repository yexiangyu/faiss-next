use bindgen::Builder;
use std::{env::var, path::PathBuf};

fn main() {
    // ignore build when detect run docs build on docs.rs
    if var("DOCS_RS").is_ok() {
        return;
    }

    // get faiss dir
    println!("cargo:rerun-if-env-changed=FAISS_DIR");
    let faiss_dir = faiss_dir();

    //bindgen
    if cfg!(feature = "bindgen") {
        faiss_bindgen()
    }

    if let Some(faiss_dir) = &faiss_dir {
        println!(
            "cargo:rustc-link-search={}",
            faiss_dir.join("lib").display()
        );
    }

    println!("cargo:rustc-link-lib=faiss_c");
    println!("cargo:rustc-link-lib=faiss");
}

fn faiss_dir() -> Option<PathBuf> {
    if let Ok(faiss_dir) = var("FAISS_DIR") {
        return Some(PathBuf::from(faiss_dir));
    }

    if cfg!(target_family = "unix") {
        if let Ok(home) = var("HOME") {
            let home = PathBuf::from(home);
            let faiss_dir = home.join("faiss");
            if faiss_dir.is_dir()
                && faiss_dir.exists()
                && faiss_dir.join("lib").exists()
                && faiss_dir.join("include").exists()
            {
                return Some(faiss_dir);
            }
        }
    } else if cfg!(target_family = "windows") {
        if let Ok(home) = var("USERPROFILE") {
            let home = PathBuf::from(home);
            let faiss_dir = home.join("faiss");
            if faiss_dir.is_dir()
                && faiss_dir.exists()
                && faiss_dir.join("lib").exists()
                && faiss_dir.join("include").exists()
            {
                return Some(faiss_dir);
            }
        }

        let tool = PathBuf::from("c:\\tools\\faiss");
        if tool.exists()
            && tool.is_dir()
            && tool.join("lib").exists()
            && tool.join("include").exists()
        {
            return Some(tool);
        }
    }

    None
}

fn faiss_bindgen() {
    println!("cargo:rerun-if-changed=faiss.h");

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

    if let Some(faiss_dir) = faiss_dir() {
        builder = builder.clang_arg(format!("-I{}", faiss_dir.join("include").display()));
    }

    let os_name = var("CARGO_CFG_TARGET_OS").expect("no in cargo");
    let output_dir = PathBuf::from("src").join(os_name);

    std::fs::create_dir_all(&output_dir).expect("failed to create output_dir");

    let mut output = output_dir.join("bindings.rs");

    if cfg!(feature = "gpu") {
        output = output_dir.join("bindings_gpu.rs");
        builder = builder
            .clang_arg("-DFAISS_USE_GPU")
            .clang_arg(format!("-I{}", get_cuda_include_dir()));
    }

    builder
        .generate()
        .expect("unable to generate bindings")
        .write_to_file(&mut output)
        .expect("unable to write bindings");
}

fn get_cuda_include_dir() -> String {
    let mut inc_dir = "".to_string();

    if cfg!(target_os = "linux") {
        inc_dir = "/usr/local/cuda/include/".to_string();
    } else if cfg!(target_os = "windows") {
        let cuda_path = var("CUDA_PATH").expect("failed to find cuda path");
        inc_dir = format!("{}", PathBuf::from(cuda_path).join("include").display());
    }

    if inc_dir.is_empty() {
        panic!("could not find cuda include dir");
    }

    inc_dir
}
