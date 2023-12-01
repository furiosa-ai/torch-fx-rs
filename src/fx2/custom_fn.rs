use std::{fmt, sync::Arc};

use pyo3::{
    pyclass, pymethods,
    types::{PyDict, PyTuple},
    PyObject, PyResult,
};

/// Wrapper for a Rust function. This wraps a function to execute it in Python.
/// Therefore, the function needs to
/// receive 2 arguments, args as `&PyTuple` and kwargs as `Option<&PyDict>`,
/// and return `PyResult<PyObject>`.
pub type FunctionWrapper =
    Arc<dyn Fn(&PyTuple, Option<&PyDict>) -> PyResult<PyObject> + Sync + Send>;

/// An interface for Python callable object which actually executes a Rust function.
#[pyclass]
#[derive(Clone)]
pub struct CustomFn {
    /// Name of the custom function
    pub func_name: String,
    func: FunctionWrapper,
}

impl fmt::Debug for CustomFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CustomFn({})", self.func_name)
    }
}

#[pymethods]
impl CustomFn {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(&self, args: &PyTuple, kwargs: Option<&PyDict>) -> PyResult<PyObject> {
        (*self.func)(args, kwargs)
    }

    #[getter]
    fn __name__(&self) -> PyResult<&str> {
        Ok(&self.func_name)
    }
}

impl CustomFn {
    /// Create a new Python callable object
    /// which is named as the value of `func_name`
    /// and actually executes a Rust function wrapped in `func`.
    pub fn new<S: AsRef<str>>(func_name: S, func: FunctionWrapper) -> Self {
        Self {
            func_name: func_name.as_ref().to_string(),
            func,
        }
    }
}
