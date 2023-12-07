use std::{
    collections::HashMap,
    fmt::{self, Error},
    ops::Deref,
};

use indexmap::IndexMap;
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError},
    types::{PyDict, PyIterator, PyList, PyTuple},
    AsPyPointer, FromPyObject, IntoPy, Py, PyAny, PyNativeType, PyObject, PyResult, PyTypeInfo,
    Python, ToPyObject,
};

use crate::fx::{custom_fn::CustomFn, node::Node, Argument, Op, Target};

/// A wrapper for PyTorch's [`Graph`][graph] class.
///
/// The constructor method of this returns a shared reference `&Graph`
/// instead of an owned value. The return value is GIL-bound owning
/// reference into Python's heap.
///
/// [graph]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph
#[repr(transparent)]
pub struct Graph(PyAny);

unsafe impl PyNativeType for Graph {}

impl ToPyObject for Graph {
    #[inline]
    fn to_object(&self, py: Python<'_>) -> PyObject {
        unsafe { PyObject::from_borrowed_ptr(py, self.as_ptr()) }
    }
}

impl AsRef<PyAny> for Graph {
    #[inline]
    fn as_ref(&self) -> &PyAny {
        &self.0
    }
}

impl Deref for Graph {
    type Target = PyAny;

    #[inline]
    fn deref(&self) -> &PyAny {
        &self.0
    }
}

impl AsPyPointer for Graph {
    #[inline]
    fn as_ptr(&self) -> *mut pyo3::ffi::PyObject {
        self.0.as_ptr()
    }
}

impl IntoPy<Py<Graph>> for &'_ Graph {
    #[inline]
    fn into_py(self, py: Python<'_>) -> Py<Graph> {
        unsafe { Py::from_borrowed_ptr(py, self.as_ptr()) }
    }
}

impl From<&'_ Graph> for Py<Graph> {
    #[inline]
    fn from(value: &'_ Graph) -> Self {
        unsafe { Py::from_borrowed_ptr(value.py(), value.as_ptr()) }
    }
}

impl<'a> From<&'a Graph> for &'a PyAny {
    #[inline]
    fn from(value: &'a Graph) -> Self {
        unsafe { &*(value as *const Graph as *const PyAny) }
    }
}

unsafe impl PyTypeInfo for Graph {
    type AsRefTarget = Self;

    const NAME: &'static str = "Graph";
    const MODULE: Option<&'static str> = Some("torch.fx");

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut pyo3::ffi::PyTypeObject {
        PyAny::type_object_raw(py)
    }
}

impl<'py> FromPyObject<'py> for &'py Graph {
    #[inline]
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        ob.downcast().map_err(Into::into)
    }
}

impl Graph {
    /// Create new instance of `Graph` PyTorch class with PyTorch native constructor.
    ///
    /// If new instance is created successfully, returns `Ok` with a shared reference
    /// to the newly created instance in it. Otherwise, returns `Err` with a `PyErr`
    /// in it. The `PyErr` will explain the cause of the failure.
    pub fn new(py: Python<'_>) -> PyResult<&Self> {
        let module = py.import("torch.fx")?.getattr("Graph")?;
        let origin = module.getattr("__new__")?.call1((module,))?;
        module.getattr("__init__")?.call1((origin,))?;
        Ok(origin.downcast()?)
    }

    /// Retrieve all the [`Node`s][nodes] of this `Graph` as a Python iterator.
    ///
    /// If the retrieval is done successfully, returns `Ok` with a shared reference
    /// to a Python iterator for it in it. Otherwise, returns `Err` with a `PyErr`
    /// in it. The `PyErr` will explain the cause of the failure.
    ///
    /// [nodes]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.nodes
    pub fn nodes(&self) -> PyResult<&PyIterator> {
        self.getattr("nodes")?.iter()
    }

    /// An interface for [`eliminate_dead_code`][eliminate_dead_code] instance
    /// method of `Graph` PyTorch class.
    ///
    /// If the method call is done successfully, returns `Ok(())`. Otherwise,
    /// returns `Err` with a `PyErr` in it. The `PyErr` will explain the cause of
    /// the failure.
    ///
    /// [eliminate_dead_code]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.eliminate_dead_code
    pub fn eliminate_dead_code(&self) -> PyResult<()> {
        self.getattr("eliminate_dead_code")?.call0()?;
        Ok(())
    }

    /// An interface for [`lint`][lint] instance method of `Graph` PyTorch
    /// class.
    ///
    /// If the
    /// method call is done successfully, returns `Ok(())`. Otherwise, returns
    /// `Err` with a `PyErr` in it. The `PyErr` will explain the cause of the
    /// failure.
    ///
    /// [lint]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.lint
    pub fn lint(&self) -> PyResult<()> {
        self.getattr("lint")?.call0()?;
        Ok(())
    }

    /// An interface for [`create_node`][create_node] instance method of
    /// `Graph` PyTorch class, but `type_expr` is not given (`None`).
    /// Also, if `meta` is given, the newly created `Node` will have an
    /// attribute `meta`, whose value will be the given argument `meta`.
    ///
    /// If the method call is done successfully, returns
    /// `Ok` with a shared reference to the newly created `Node` in it.
    /// Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will
    /// explain the cause of the failure.
    ///
    /// [create_node]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.create_node
    pub fn create_node<S: AsRef<str>>(
        &self,
        op: Op,
        target: Target,
        args: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = Argument>>,
        kwargs: impl IntoIterator<Item = (String, Argument)>,
        name: S,
        meta: HashMap<String, PyObject>,
    ) -> PyResult<&Node> {
        let py = self.py();
        let node_args = PyTuple::new(py, &[op.into_py(py), target.into_py(py)]);
        let node_kwargs = {
            let node_kwargs = PyDict::new(py);
            node_kwargs.set_item(
                "args",
                PyTuple::new(
                    py,
                    args.into_iter()
                        .map(|arg| self.argument_into_py(py, arg))
                        .collect::<PyResult<Vec<_>>>()?,
                ),
            )?;
            node_kwargs.set_item(
                "kwargs",
                kwargs
                    .into_iter()
                    .map(|(key, arg)| Ok((key, self.argument_into_py(py, arg)?)))
                    .collect::<PyResult<HashMap<_, _>>>()?,
            )?;
            node_kwargs.set_item("name", name.as_ref())?;
            Some(node_kwargs)
        };
        let create_node_fn = self.getattr("create_node")?;
        let node = create_node_fn.call(node_args, node_kwargs)?;
        node.setattr("meta", meta)?;
        Ok(node.downcast()?)
    }

    /// Create and insert a placeholder `Node` into this `Graph`.
    /// A placeholder represents a function input.
    /// `name` is the name for the input value.
    ///
    /// This does the same what [`placeholder`][placeholder] instance method of
    /// `Graph` PyTorch class does, but `type_expr` is `None` and `default_value`
    /// is `inspect.Signature.empty`.
    ///
    /// If the creation and insertion of the `Node` is done successfully,
    /// returns `Ok` with a shared reference to the newly created `Node` in
    /// it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will
    /// explain the cause of the failure.
    ///
    /// [placeholder]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.placeholder
    pub fn new_placeholder<S: AsRef<str>>(&self, name: S) -> PyResult<&Node> {
        self.create_node(
            Op::Placeholder,
            Target::Str(name.as_ref().to_string()),
            None,
            None,
            name,
            Default::default(),
        )
    }

    /// Create and insert an output `Node` into this `Graph`.
    /// `args` is the value that should be returned by this output node.
    /// `args` has to be `Argument::NodeTuple`
    ///
    /// This does the same what [`output`][output] instance method of
    /// `Graph` PyTorch class does, but `type_expr` is `None` and
    /// the newly created `Node` has a name 'output'.
    ///
    /// If the creation and insertion of the `Node` is done successfully,
    /// returns `Ok` with a shared reference to the newly created `Node` in
    /// it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will
    /// explain the cause of the failure.
    ///
    /// [output]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.output
    pub fn new_output(&self, args: Argument) -> PyResult<&Node> {
        if !matches!(args, Argument::NodeTuple(_)) {
            return Err(PyTypeError::new_err("output arg must be a tuple of nodes"));
        }

        let name = "output";
        self.create_node(
            Op::Output,
            Target::Str(name.to_string()),
            vec![args],
            None,
            name,
            Default::default(),
        )
    }

    /// Create and insert a call_function `Node` into this `Graph`.
    /// call_function `Node` represents a call to a Python callable,
    /// specified by `custom_fn`.
    ///
    /// This does the same what [`call_function`][call_function] instance method
    /// of `Graph` PyTorch class does, but the name of `the_function` parameter
    /// is changed into `custom_fn`, `type_expr` is not given (`None`), and
    /// the `name` for the name of this node is given.
    ///
    /// If the creation and insertion of the `Node` is done successfully,
    /// returns `Ok` with a shared reference to the newly created `Node` in
    /// it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will
    /// explain the cause of the failure.
    ///
    /// [call_function]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.call_function
    pub fn new_custom_fn<S: AsRef<str>>(
        &self,
        name: S,
        args: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = Argument>>,
        kwargs: impl IntoIterator<Item = (String, Argument)>,
        custom_fn: CustomFn,
    ) -> PyResult<&Node> {
        self.create_node(
            Op::CallFunction,
            Target::CustomFn(custom_fn),
            args,
            kwargs,
            name.as_ref(),
            Default::default(),
        )
    }

    /// Copy a `Node` from another `Graph` into this `Graph`(`self`).
    /// `node` is the node to copy into `self`.
    /// `mapper` needs to transform arguments
    /// from the graph of `node` to the graph of self.
    ///
    /// This does the same what [`node_copy`][node_copy] instance method of `Graph`
    /// PyTorch class does.
    ///
    /// If the copying and insertion of the `Node` is done successfuly,
    /// returns `Ok` with a shared reference to the newly created `Node` in
    /// it. Otherwise, returns `Err` with a `PyErr` in it. The `PyErr` will
    /// explain the cause of the failure.
    ///
    /// [node_copy]: https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.node_copy
    pub fn copy_node(
        &self,
        node: &Node,
        mapper: Option<&HashMap<String, String>>,
    ) -> PyResult<&Node> {
        let mut args = node.args()?;
        if let Some(mapper) = mapper {
            args = args
                .iter()
                .map(|arg| match arg {
                    Argument::Node(name) => {
                        let mapped = mapper.get(name).ok_or(PyRuntimeError::new_err(format!(
                            "Failed to get mapped arg from mapper {name:?}",
                        )))?;
                        Ok(Argument::Node(mapped.clone()))
                    }
                    Argument::NodeTuple(names) => {
                        let mapped = names
                            .iter()
                            .map(|name| {
                                let mapped = mapper.get(name).ok_or(PyRuntimeError::new_err(
                                    format!("Failed to get mapped arg from mapper {name:?}"),
                                ))?;
                                Ok(mapped.clone())
                            })
                            .collect::<PyResult<Vec<_>>>()?;
                        Ok(Argument::NodeTuple(mapped))
                    }
                    Argument::NodeList(names) => {
                        let mapped = names
                            .iter()
                            .map(|name| {
                                let mapped = mapper.get(name).ok_or(PyRuntimeError::new_err(
                                    format!("Failed to get mapped arg from mapper {name:?}"),
                                ))?;
                                Ok(mapped.clone())
                            })
                            .collect::<PyResult<Vec<_>>>()?;
                        Ok(Argument::NodeList(mapped))
                    }
                    others => Ok(others.clone()),
                })
                .collect::<PyResult<Vec<Argument>>>()?;
        }
        self.create_node(
            node.op()?,
            node.target()?,
            args,
            node.kwargs()?,
            node.name()?,
            node.meta()?,
        )
    }

    /// Retrieve the names of argument `Node`s of the `Node` named
    /// as the value of `node_name` in this `Graph`.
    ///
    /// If this graph doesn't have a `Node` named as the value of
    /// `node_name`, returns `Ok(None)`.
    /// If this graph have a `Node` named as the value of `node_name`,
    /// returns `Ok(Some)` with a `Vec` of names of argument `Node`s of the
    /// `Node`, in the `Some`.
    /// If something fails while looking into this `Graph`, returns `Err`
    /// with a `PyErr` in it. The `PyErr` will explain the cause of the failure.
    pub fn flatten_node_args<S: AsRef<str>>(&self, node_name: S) -> PyResult<Option<Vec<String>>> {
        let named_nodes = self.named_nodes()?;
        match named_nodes.get(node_name.as_ref()) {
            Some(node) => Ok(Some(node.downcast::<Node>()?.flatten_node_args()?)),
            None => Ok(None),
        }
    }

    /// Retrieve the names of user `Node`s of the `Node` named
    /// as the value of `node_name` in this `Graph`.
    ///
    /// If this graph doesn't have a `Node` named as the value of
    /// `node_name`, returns `Ok(None)`.
    /// If this graph have a `Node` named as the value of `node_name`,
    /// returns `Ok(Some)` with a `Vec` of names of user `Node`s of the
    /// `Node`, in the `Some`.
    /// If something fails while looking into this `Graph`, returns `Err`
    /// with a `PyErr` in it. The `PyErr` will explain the cause of the failure.
    pub fn users<S: AsRef<str>>(&self, node_name: S) -> PyResult<Option<Vec<String>>> {
        let named_nodes = self.named_nodes()?;
        match named_nodes.get(node_name.as_ref()) {
            Some(node) => {
                let user_keys = node.getattr("users")?.getattr("keys")?.call0()?;
                Ok(Some(
                    user_keys
                        .iter()?
                        .map(|r| r?.getattr("name")?.extract())
                        .collect::<PyResult<Vec<String>>>()?,
                ))
            }
            None => Ok(None),
        }
    }

    /// Stringify this `Graph`.
    ///
    /// This does the same what `__str__` instance method of `Graph`
    /// PyTorch class.
    ///
    /// If stringifying is done successfully, returns `Ok` with the
    /// resulting string in it. Otherwise, returns `Err` with a
    /// `PyErr` in it. The `PyErr` will explain the cause of the failure.
    pub fn graph_to_string(&self, py: Python<'_>) -> PyResult<String> {
        let builtin_str_fn = py.import("builtins")?.getattr("str")?;
        builtin_str_fn.call1((self,))?.extract()
    }

    /// Collect all named `Node`s of this `Graph`.
    ///
    /// Make an `IndexMap` which maps each `Node`'s name to a shared reference
    /// of the `Node` itself, for every `Node` in `self`.
    ///
    /// If this process is done successfully, returns `Ok` with the `IndexMap` in it.
    /// Otherwise, return `Err` with a `PyErr` in it.
    /// `PyErr` will explain the cause of the failure.
    pub fn named_nodes(&self) -> PyResult<IndexMap<String, &Node>> {
        self.nodes()?
            .map(|r| {
                let r = r?;
                let name: String = r.getattr("name")?.extract()?;
                Ok((name, r.downcast()?))
            })
            .collect()
    }

    /// Lookup a `Node` by its name(`name`) in this `Graph`.
    ///
    /// If there is no `Node` with a name named as the value of `name`,
    /// `Ok(None)` is returned.
    /// If there exists such `Node` in this `Graph`,
    /// `Ok(Some)` with a shared reference to the `Node` is returned.
    /// If this process fails, returns `Err` with a `PyErr` in it.
    /// `PyErr` will explain the cause of the failure.
    pub fn lookup_node<S: AsRef<str>>(&self, name: S) -> PyResult<Option<&Node>> {
        let mut nodes = self.nodes()?;
        nodes
            .find_map(|node| {
                (|| {
                    let node = node?;
                    let n: String = node.getattr("name")?.extract()?;
                    Ok(if n == name.as_ref() {
                        Some(node.downcast::<Node>()?)
                    } else {
                        None
                    })
                })()
                .transpose()
            })
            .transpose()
    }

    fn argument_into_py(&self, py: Python<'_>, arg: Argument) -> PyResult<PyObject> {
        match arg {
            Argument::Node(node_name) => {
                let node = self.lookup_node(node_name)?.unwrap();
                Ok(node.into_py(py))
            }
            Argument::NodeTuple(node_names) => Ok(PyTuple::new(py, {
                let named_nodes = self.named_nodes()?;
                node_names
                    .into_iter()
                    .map(move |name| *named_nodes.get(&name).unwrap())
            })
            .into_py(py)),
            Argument::NodeList(node_names) => Ok(PyList::new(py, {
                let named_nodes = self.named_nodes()?;
                node_names
                    .into_iter()
                    .map(move |name| *named_nodes.get(&name).unwrap())
            })
            .into_py(py)),
            Argument::OptionalNodeTuple(node_names) => Ok(PyTuple::new(py, {
                let named_nodes = self.named_nodes()?;
                node_names.into_iter().map(move |name| {
                    name.map_or_else(
                        || py.None(),
                        |name| named_nodes.get(&name).unwrap().into_py(py),
                    )
                })
            })
            .into_py(py)),
            Argument::OptionalNodeList(node_names) => Ok(PyList::new(py, {
                let named_nodes = self.named_nodes()?;
                node_names.into_iter().map(move |name| {
                    name.map_or_else(
                        || py.None(),
                        |name| named_nodes.get(&name).unwrap().into_py(py),
                    )
                })
            })
            .into_py(py)),
            Argument::NoneTuple(len) => Ok(PyTuple::new(py, vec![(); len]).into()),
            Argument::NoneList(len) => Ok(PyList::new(py, vec![(); len]).into()),
            Argument::Bool(value) => Ok(value.into_py(py)),
            Argument::Int(value) => Ok(value.into_py(py)),
            Argument::Float(value) => Ok(value.into_py(py)),
            Argument::VecBool(values) => Ok(PyList::new(py, values).into()),
            Argument::VecInt(values) => Ok(PyList::new(py, values).into()),
            Argument::VecFloat(values) => Ok(PyList::new(py, values).into()),
            Argument::Dtype(value) => Ok(value.into_py(py)),
            Argument::Layout(value) => Ok(value.into_py(py)),
            Argument::Device(value) => Ok(value.into_py(py)),
            Argument::MemoryFormat(value) => Ok(value.into_py(py)),
            Argument::Value(ob) => Ok(ob.into_py(py)),
            Argument::EmptyList => Ok(PyList::empty(py).into_py(py)),
            Argument::None => Ok(py.None()),
        }
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[derive(Debug)]
        #[allow(unused)]
        struct Graph<'a> {
            origin: PyObject,
            named_nodes: IndexMap<String, &'a Node>,
        }

        write!(
            f,
            "{:?}",
            Graph {
                origin: PyObject::from(self),
                named_nodes: self.named_nodes().map_err(|_| Error)?
            }
        )
    }
}
