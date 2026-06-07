//! `fusekit py-probe`: hidden subcommand. Verifies that the embedded
//! Python interpreter can be initialised and that `mlx.core` imports.
//!
//! This exists so we catch dyld/rpath issues with the bundled
//! cpython-3.11 before the full loader migration lands. Delete the
//! command (and this file) once the migration is stable.

use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::PyList;

pub fn run() -> Result<()> {
    // The bundled cpython-3.11 was built with a placeholder install
    // prefix (`/install/...`) so PYTHONHOME has to be steered at the
    // actual bundle root before the interpreter initialises, else
    // it can't find its stdlib (`No module named 'encodings'`).
    if std::env::var_os("PYTHONHOME").is_none() {
        std::env::set_var(
            "PYTHONHOME",
            "/Applications/oMLX.app/Contents/Python/cpython-3.11",
        );
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| -> PyResult<()> {
        let sys = py.import_bound("sys")?;
        let version: String = sys.getattr("version")?.extract()?;
        println!(
            "[py-probe] embedded python: {}",
            version.split_whitespace().next().unwrap_or("?")
        );

        // Inject the oMLX site-packages so `import mlx.core` works.
        let sp = "/Applications/oMLX.app/Contents/Python/framework-mlx-framework/lib/python3.11/site-packages";
        let path: Bound<PyList> = sys.getattr("path")?.downcast_into()?;
        path.insert(0, sp)?;

        let mx = py.import_bound("mlx.core")?;
        let v: String = mx.getattr("__version__")?.extract()?;
        println!("[py-probe] mlx.core: {v}");

        let _ = py.import_bound("mlx_lm")?;
        println!("[py-probe] mlx_lm: ok");
        Ok(())
    })?;
    Ok(())
}
