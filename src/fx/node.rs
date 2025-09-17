use std::{
    collections::HashMap,
    fmt::{self, Error},
    ops::Deref,
};

use pyo3::{
    types::PyDict, AsPyPointer, FromPyObject, IntoPy, Py, PyAny, PyNativeType, PyObject, PyResult,
    PyTypeInfo, Python, ToPyObject,
};

use crate::fx::{Argument, Op, Target, TensorMeta};

/// A wrapper for PyTorch's [`Node`][node] class.
///
/// This appears as a shared reference `&Node` into Python's heap
/// instead of an owned value.
///
/// [node]: https://pytorch.org/docs/stable/fx.html#torch.fx.Node
#[repr(transparent)]
pub struct Node(PyAny);

unsafe impl PyNativeType for Node {}

impl ToPyObject for Node {
    #[inline]
    fn to_object(&self, py: Python<'_>) -> PyObject {
        unsafe { PyObject::from_borrowed_ptr(py, self.as_ptr()) }
    }
}

impl AsRef<PyAny> for Node {
    #[inline]
    fn as_ref(&self) -> &PyAny {
        &self.0
    }
}

impl Deref for Node {
    type Target = PyAny;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsPyPointer for Node {
    #[inline]
    fn as_ptr(&self) -> *mut pyo3::ffi::PyObject {
        self.0.as_ptr()
    }
}

impl IntoPy<Py<Node>> for &'_ Node {
    #[inline]
    fn into_py(self, py: Python<'_>) -> Py<Node> {
        unsafe { Py::from_borrowed_ptr(py, self.as_ptr()) }
    }
}

impl From<&'_ Node> for Py<Node> {
    #[inline]
    fn from(value: &'_ Node) -> Self {
        unsafe { Py::from_borrowed_ptr(value.py(), value.as_ptr()) }
    }
}

impl<'a> From<&'a Node> for &'a PyAny {
    #[inline]
    fn from(value: &'a Node) -> Self {
        unsafe { &*(value as *const Node as *const PyAny) }
    }
}

unsafe impl PyTypeInfo for Node {
    type AsRefTarget = Self;

    const NAME: &'static str = "Node";
    const MODULE: Option<&'static str> = Some("torch.fx");

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut pyo3::ffi::PyTypeObject {
        PyAny::type_object_raw(py)
    }
}

impl<'py> FromPyObject<'py> for &'py Node {
    #[inline]
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        ob.downcast().map_err(Into::into)
    }
}

type MetaTuple = (HashMap<String, PyObject>, Option<Vec<TensorMeta>>);

impl Node {
    /// Retrieve the names of argument `Node`s of this `Node`.
    /// Although a `Node` can have multiple arguments and
    /// an argument can have one or more `Node`s,
    /// the result will contain all the argument `Node`s' names
    /// in a 1-dimensional vector.
    /// (This is why this method is named `flatten_node_args`.)
    ///
    /// If the retrieval is done successfully, returns `Ok` with a `Vec`
    /// of names of argument nodes.
    /// Otherwise, returns `Err` with a `PyErr` in it.
    /// The `PyErr` will explain the cause of the failure.
    pub fn flatten_node_args(&self) -> PyResult<Vec<String>> {
        let mut flatten_args = vec![];
        let args = self.getattr("args")?.iter()?;
        for obj in args {
            let arg = Argument::extract(obj?)?;
            match arg {
                Argument::Node(name) => flatten_args.push(name),
                Argument::NodeList(nl) | Argument::NodeTuple(nl) => flatten_args.extend(nl),
                _ => (),
            }
        }
        Ok(flatten_args)
    }

    /// Retrieve the arguments of this `Node`.
    ///
    /// If the retrieval is done successfully, returns `Ok`
    /// with a `Vec<Argument>` containing the arguments.
    /// Otherwise, returns `Err` with a `PyErr` in it.
    /// The `PyErr` will explain the cause of the failure.
    pub fn args(&self) -> PyResult<Vec<Argument>> {
        self.getattr("args")?.extract()
    }

    /// Retrieve the name of this `Node`.
    ///
    /// If the retrieval is done successfully, returns `Ok`
    /// with the name in it.
    /// Otherwise, returns `Err` with a `PyErr` in it.
    /// The `PyErr` will explain the cause of the failure.
    pub fn name(&self) -> PyResult<String> {
        self.getattr("name")?.extract()
    }

    /// Retrieve the opcode of this `Node`.
    ///
    /// If the retrieval is done successfully, returns `Ok`
    /// with the opcode in `Op` in it.
    /// Otherwise, returns `Err` with a `PyErr` in it.
    /// The `PyErr` will explain the cause of the failure.
    pub fn op(&self) -> PyResult<Op> {
        self.getattr("op")?.extract()
    }

    /// Retrieve the target this `Node` should call.
    ///
    /// If the retrieval is done successfully, returns `Ok`
    /// with the target in `Target` in it.
    /// Otherwise, returns `Err` with a `PyErr` in it.
    /// The `PyErr` will explain the cause of the failure.
    pub fn target(&self) -> PyResult<Target> {
        self.getattr("target")?.extract()
    }

    /// Retrieve the kwargs to be passed to the target of this `Node`.
    ///
    /// If the retrieval is done successfully, returns `Ok`
    /// with the kwargs in `HashMap<String, Argument>` in it.
    /// Otherwise, returns `Err` with a `PyErr` in it.
    /// The `PyErr` will explain the cause of the failure.
    pub fn kwargs(&self) -> PyResult<HashMap<String, Argument>> {
        self.getattr("kwargs")?.extract()
    }

    /// Retrieve the meta of this `Node`.
    ///
    /// If this `Node` has an attribute `meta`, returns `Ok`
    /// with the meta in `HashMap<String, PyObject>` in it.
    /// Otherwise, returns `Ok(Default::default())`.
    /// This never returns `Err`.
    pub fn meta(&self) -> PyResult<HashMap<String, PyObject>> {
        if let Ok(meta) = self.getattr("meta") {
            meta.extract()
        } else {
            Ok(Default::default())
        }
    }

    /// Retrieve the meta dictionary as a Python mapping without copying.
    ///
    /// Returns `Ok(Some(&PyDict))` if `self.meta` exists and is a dict,
    /// `Ok(None)` if no `meta` attribute exists.
    pub fn meta_pydict(&self) -> PyResult<Option<&PyDict>> {
        if let Ok(meta) = self.getattr("meta") {
            Ok(Some(meta.downcast::<PyDict>()?))
        } else {
            Ok(None)
        }
    }

    fn extract_meta_tensor_meta(&self) -> PyResult<MetaTuple> {
        Ok(if let Ok(meta) = self.getattr("meta") {
            let tensor_meta = meta.get_item("tensor_meta").ok().map(TensorMeta::extracts_tensor_meta).transpose().unwrap_or_else(|e| {
                tracing::debug!("Failed to extract tensor_meta, probably it is gm before 'shape_prop' called., {e:?}");
                None
            });
            let meta = meta.extract()?;
            (meta, tensor_meta)
        } else {
            (Default::default(), Default::default())
        })
    }

    fn extract_users(&self) -> PyResult<Vec<String>> {
        let user_keys = self.getattr("users")?.getattr("keys")?.call0()?;
        user_keys
            .iter()?
            .map(|r| r?.getattr("name")?.extract())
            .collect()
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (|| -> PyResult<fmt::Result> {
            let (meta, tensor_meta) = self.extract_meta_tensor_meta()?;
            Ok(f.debug_struct("Node")
                .field("origin", &PyObject::from(self))
                .field("name", &self.name()?)
                .field("op", &self.op()?)
                .field("target", &self.target())
                .field("args", &self.args())
                .field("kwargs", &self.kwargs()?)
                .field(
                    "stack_trace",
                    &self.getattr("stack_trace")?.extract::<Option<PyObject>>()?,
                )
                .field("meta", &meta)
                .field("tensor_meta", &tensor_meta)
                .field("users", &self.extract_users()?)
                .finish())
        })()
        .map_err(|_| Error)?
    }
}
