use anyhow::Result;

fn main() -> Result<()> {
    if cfg!(feature = "bindgen") {
        gen()?;
    }
    link()?;
    Ok(())
}

fn link() -> Result<()> {
    println!("cargo:rustc-link-lib=faiss_c");
    println!("cargo:rustc-link-lib=faiss");
    Ok(())
}

fn gen() -> Result<()> {
    println!("cargo:rerun-if-changed=faiss.h");
    println!("cargo:rerun-if-env-changed=FAISS_INCLUDE_DIR");

    let mut builder = bindgen::Builder::default()
        .header("faiss.h")
        .derive_default(true)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        .size_t_is_usize(true)
        .layout_tests(false)
        .allowlist_function("faiss_.*")
        .allowlist_type("idx_t|Faiss.*")
        .opaque_type("FILE");

    if let Some(inc_dir) = get_include_dir() {
        builder = builder.clang_arg(format!("-I{}", inc_dir));
    }

    if cfg!(feature = "gpu") {
        builder = builder.clang_arg("-DFAISS_USE_GPU");
        if let Some(cuda_dir) = get_cuda_dir() {
            builder = builder.clang_arg(format!("-I{}/include", cuda_dir));
        }
    }

    let bindings = builder.generate()?;

    let tpl = triplet();

    bindings.write_to_file(format!("src/{}.rs", &tpl))?;

    Ok(())
}

fn get_include_dir() -> Option<String> {
    if let Ok(inc_dir) = std::env::var("FAISS_INCLUDE_DIR") {
        return Some(inc_dir);
    }

    if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64")
        // Apple Silicon
        {
            let inc_dir = "/opt/homebrew/opt/faiss/include";
            if std::path::Path::new(inc_dir).exists() {
                return Some(inc_dir.to_string());
            }
        } else if cfg!(target_arch = "x64_64")
        // Intel
        {
            let inc_dir = "/usr/local/homebrew/opt/faiss/include";
            if std::path::Path::new(inc_dir).exists() {
                return Some(inc_dir.to_string());
            }
        }
    } else if cfg!(target_os = "linux") {
        return None;
    } else {
        panic!("os not supported");
    };
    unreachable!()
}

fn get_cuda_dir() -> Option<String> {
    if cfg!(target_os = "linux") {
        let cuda_dir = "/usr/local/cuda";
        if std::path::PathBuf::from(cuda_dir).exists() {
            return Some(cuda_dir.to_string());
        }
        return None;
    }
    unreachable!()
}

fn triplet() -> String {
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        panic!("os not supported");
    };

    let arch = if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else {
        panic!("arch not supported");
    };

    let use_gpu = if cfg!(feature = "gpu") { "gpu" } else { "cpu" };

    format!("{}-{}-{}", os, arch, use_gpu)
}
