//! Build script for fusekit.
//!
//! Two responsibilities:
//!   1. Wire pyo3 (embedded cpython-3.11) to the oMLX-bundled libpython
//!      so the patch-time graph capture can run mlx_lm in-process.
//!   2. Compile our `metal/*.metal` kernels into a single `metallib`
//!      that gets embedded in the fusekit binary.
//!   3. Build the `fusekit_runtime_native` cdylib (separate pyo3
//!      extension-module) and embed its `.so` via `include_bytes!`.

use std::path::PathBuf;

fn main() {
    // The bundled libpython has its install_name baked to a fake
    // `/install/lib/...` build prefix; the new Apple linker dropped
    // `-Wl,-change`, so we copy the dylib into OUT_DIR and rewrite the
    // copy's id to `@rpath/libpython3.11.dylib`. At runtime dyld looks
    // for that @rpath which we point at the bundle.
    let py_root = std::path::Path::new("/Applications/oMLX.app/Contents/Python/cpython-3.11");
    let py_lib_src = py_root.join("lib").join("libpython3.11.dylib");
    if py_lib_src.is_file() {
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR from cargo");
        let fixed = std::path::Path::new(&out_dir).join("libpython3.11.dylib");
        std::fs::copy(&py_lib_src, &fixed).expect("copy libpython3.11.dylib to OUT_DIR");
        let status = std::process::Command::new("install_name_tool")
            .args(["-id", "@rpath/libpython3.11.dylib", fixed.to_str().unwrap()])
            .status()
            .expect("invoke install_name_tool");
        assert!(status.success(), "install_name_tool -id failed");

        println!("cargo:rustc-link-search=native={}", out_dir);
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            py_root.join("lib").display()
        );
    } else {
        panic!(
            "expected oMLX-bundled libpython at {} — is oMLX.app installed?",
            py_lib_src.display()
        );
    }

    // Align with the oMLX-bundled dylibs' deployment target.
    std::env::set_var("MACOSX_DEPLOYMENT_TARGET", "26.0");
    println!("cargo:rustc-link-arg=-mmacosx-version-min=26.0");

    for framework in ["Metal", "Foundation", "QuartzCore"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }

    compile_metal_kernels();
    build_python_runtime_native();
}

/// Build the `fusekit_runtime_native` cdylib (pyo3 extension) and
/// copy it to `OUT_DIR/_fusekit_runtime_native.so`. The fusekit
/// binary embeds that file via `include_bytes!` from
/// `runtime_payload.rs` and extracts it to each model's
/// `fusekit/runtime/fusekit_runtime/` directory at patch time.
///
/// We invoke cargo from build.rs so the cdylib gets its own build
/// pipeline (extension-module feature, dynamic-lookup linker flag),
/// independent of fusekit's own Metal link config. To avoid lock
/// contention with the outer cargo invocation, the nested build uses
/// a *separate* `CARGO_TARGET_DIR` under `OUT_DIR`.
fn build_python_runtime_native() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let crate_dir = manifest_dir.join("python_runtime");
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());

    println!(
        "cargo:rerun-if-changed={}",
        crate_dir.join("Cargo.toml").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        crate_dir.join("build.rs").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        crate_dir.join("src/lib.rs").display()
    );

    let inner_target = out_dir.join("runtime-native-target");
    std::fs::create_dir_all(&inner_target).expect("create inner cargo target dir");

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let mut args = vec!["build", "--manifest-path"];
    let manifest_path_str = crate_dir.join("Cargo.toml").to_string_lossy().into_owned();
    args.push(&manifest_path_str);
    if profile == "release" {
        args.push("--release");
    }
    let status = std::process::Command::new("cargo")
        .args(&args)
        .env("CARGO_TARGET_DIR", &inner_target)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_BUILD_JOBS")
        .status()
        .expect("invoke cargo for fusekit_runtime_native");
    assert!(status.success(), "fusekit_runtime_native build failed");

    let src = inner_target
        .join(&profile)
        .join("lib_fusekit_runtime_native.dylib");
    let dst = out_dir.join("_fusekit_runtime_native.so");
    std::fs::copy(&src, &dst)
        .unwrap_or_else(|e| panic!("copy {} -> {}: {e}", src.display(), dst.display()));
    println!(
        "cargo:rustc-env=FUSEKIT_RUNTIME_NATIVE_SO={}",
        dst.display()
    );
}

/// Compile all `metal/*.metal` files into a single `fusekit_kernels.metallib`
/// next to the binary. The dispatcher loads this at runtime.
fn compile_metal_kernels() {
    let metal_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("metal");
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());

    let metal_bin = match std::process::Command::new("xcrun")
        .args(["--sdk", "macosx", "--find", "metal"])
        .output()
    {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => {
            println!("cargo:warning=Metal toolchain not found via xcrun — skipping kernel build");
            return;
        }
    };
    let metallib_bin = std::process::Command::new("xcrun")
        .args(["--sdk", "macosx", "--find", "metallib"])
        .output()
        .ok()
        .and_then(|o| {
            o.status
                .success()
                .then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
        });

    let Ok(entries) = std::fs::read_dir(&metal_dir) else {
        println!("cargo:warning=no metal/ directory — skipping kernel build");
        return;
    };

    let mut air_files = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("metal") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let air = out_dir.join(format!("{stem}.air"));
        let status = std::process::Command::new(&metal_bin)
            .args(["-c", "-O3", "-ffast-math"])
            .arg(&path)
            .arg("-o")
            .arg(&air)
            .status()
            .expect("invoke metal compiler");
        assert!(
            status.success(),
            "metal compile failed for {}",
            path.display()
        );
        air_files.push(air);
    }

    if air_files.is_empty() {
        return;
    }
    let lib_path = out_dir.join("fusekit_kernels.metallib");
    if let Some(metallib_bin) = metallib_bin {
        let status = std::process::Command::new(&metallib_bin)
            .args(&air_files)
            .arg("-o")
            .arg(&lib_path)
            .status()
            .expect("invoke metallib linker");
        assert!(status.success(), "metallib link failed");
    } else {
        let status = std::process::Command::new(&metal_bin)
            .args(&air_files)
            .arg("-o")
            .arg(&lib_path)
            .status()
            .expect("invoke metal as linker");
        assert!(status.success(), "metal link to metallib failed");
    }

    println!(
        "cargo:rustc-env=FUSEKIT_KERNELS_METALLIB={}",
        lib_path.display()
    );
    println!("cargo:rerun-if-changed={}", metal_dir.display());
}
