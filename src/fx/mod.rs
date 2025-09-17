mod custom_fn;
mod graph;
mod graph_module;
mod node;
mod types {
    use pyo3::{
        exceptions::PyTypeError,
        types::{PyList, PyString, PyTuple},
        FromPyObject, IntoPy, PyAny, PyErr, PyObject, PyResult, Python,
    };
    use std::collections::HashMap;

    use super::custom_fn::CustomFn;

    /// A representation of opcodes for `Node`s.
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum Op {
        Placeholder,
        CallFunction,
        CallMethod,
        CallModule,
        GetAttr,
        Output,
    }

    impl<'py> FromPyObject<'py> for Op {
        fn extract(op: &'py PyAny) -> PyResult<Self> {
            let op_str: &str = op.extract()?;
            match op_str {
                "placeholder" => Ok(Self::Placeholder),
                "get_attr" => Ok(Self::GetAttr),
                "call_function" => Ok(Self::CallFunction),
                "call_method" => Ok(Self::CallMethod),
                "call_module" => Ok(Self::CallModule),
                "output" => Ok(Self::Output),
                _ => Err(PyErr::new::<PyTypeError, _>(format!(
                    "unsupported op type, {op_str}"
                ))),
            }
        }
    }

    impl IntoPy<PyObject> for Op {
        fn into_py(self, py: Python<'_>) -> PyObject {
            match self {
                Op::Placeholder => PyString::new(py, "placeholder").into(),
                Op::CallFunction => PyString::new(py, "call_function").into(),
                Op::CallMethod => PyString::new(py, "call_method").into(),
                Op::CallModule => PyString::new(py, "call_module").into(),
                Op::GetAttr => PyString::new(py, "get_attr").into(),
                Op::Output => PyString::new(py, "output").into(),
            }
        }
    }

    /// A representation of targets for `Node`s.
    #[derive(Debug, Clone)]
    pub enum Target {
        Str(String),
        TorchOp(String, PyObject),
        BuiltinFn(String, PyObject),
        Callable(PyObject),
        CustomFn(CustomFn),
    }

    impl<'py> FromPyObject<'py> for Target {
        fn extract(target: &'py PyAny) -> PyResult<Self> {
            let clsname: String = target
                .getattr("__class__")?
                .getattr("__name__")?
                .extract()?;
            match clsname.as_str() {
                "OpOverload" => {
                    let name = target.getattr("_name")?.extract()?;
                    Ok(Self::TorchOp(name, target.into()))
                }
                "builtin_function_or_method" => {
                    let name = target.getattr("__name__")?.extract()?;
                    Ok(Self::BuiltinFn(name, target.into()))
                }
                "str" => Ok(Self::Str(target.extract()?)),
                _ => {
                    // TODO, handle custom fn
                    if target.is_callable() {
                        Ok(Target::Callable(target.into()))
                    } else {
                        Err(PyErr::new::<PyTypeError, _>(format!(
                            "Unsupported class of target, and not callable, cls: {clsname}",
                        )))
                    }
                }
            }
        }
    }

    impl IntoPy<PyObject> for Target {
        fn into_py(self, py: Python<'_>) -> PyObject {
            match self {
                Target::Str(s) => PyString::new(py, s.as_str()).into(),
                Target::TorchOp(_, ob) => ob.into_py(py),
                Target::BuiltinFn(_, ob) => ob.into_py(py),
                Target::Callable(ob) => ob.into_py(py),
                Target::CustomFn(f) => f.into_py(py),
            }
        }
    }

    /// A representation of arguments for `Node`s.
    #[derive(Debug, Clone)]
    pub enum Argument {
        Node(String),
        NodeList(Vec<String>),
        NodeTuple(Vec<String>),
        OptionalNodeList(Vec<Option<String>>),
        OptionalNodeTuple(Vec<Option<String>>),
        NoneList(usize),
        NoneTuple(usize),
        Bool(bool),
        Int(i64),
        Float(f64),
        VecBool(Vec<bool>),
        VecInt(Vec<i64>),
        VecFloat(Vec<f64>),
        Dtype(Dtype),
        Layout(Layout),
        Device(Device),
        MemoryFormat(MemoryFormat),
        Value(PyObject),
        EmptyList,
        None,
    }

    impl<'py> FromPyObject<'py> for Argument {
        fn extract(arg: &'py PyAny) -> PyResult<Self> {
            if let Ok(arg) = arg.downcast::<PyTuple>() {
                if let Ok(args) = arg
                    .iter()
                    .map(Self::into_node_name)
                    .collect::<PyResult<Vec<_>>>()
                {
                    if args.iter().all(|arg| arg.is_some()) {
                        let args = args.into_iter().map(Option::unwrap).collect::<Vec<_>>();
                        return Ok(Self::NodeTuple(args));
                    } else if args.iter().any(|arg| arg.is_some()) {
                        return Ok(Self::OptionalNodeTuple(args));
                    } else {
                        return Ok(Self::NoneTuple(args.len()));
                    }
                }

                return Err(PyTypeError::new_err(format!(
                    "PyTuple, but none of NodeTuple, OptionalNodeTuple, NoneTuple, {arg:?}"
                )));
            } else if let Ok(arg) = arg.downcast::<PyList>() {
                if arg.is_empty() {
                    return Ok(Self::EmptyList);
                }

                if let Ok(args) = arg
                    .iter()
                    .map(Self::into_node_name)
                    .collect::<PyResult<Vec<_>>>()
                {
                    if args.iter().all(|arg| arg.is_some()) {
                        let args = args.into_iter().map(Option::unwrap).collect::<Vec<_>>();
                        return Ok(Self::NodeList(args));
                    } else if args.iter().any(|arg| arg.is_some()) {
                        return Ok(Self::OptionalNodeList(args));
                    } else {
                        return Ok(Self::NoneList(args.len()));
                    }
                }

                if let Ok(args) = arg
                    .iter()
                    .map(|arg| arg.extract())
                    .collect::<PyResult<Vec<bool>>>()
                {
                    return Ok(Self::VecBool(args));
                }

                if let Ok(args) = arg
                    .iter()
                    .map(|arg| arg.extract())
                    .collect::<PyResult<Vec<i64>>>()
                {
                    return Ok(Self::VecInt(args));
                }

                if let Ok(args) = arg
                    .iter()
                    .map(|arg| arg.extract())
                    .collect::<PyResult<Vec<f64>>>()
                {
                    return Ok(Self::VecFloat(args));
                }

                return Err(PyTypeError::new_err(format!("PyList, but none of EmptyList, NodeList, OptionalNodeList, NoneList, VecBool, VecInt, VecFloat: {arg:?}")));
            } else if let Ok(value) = arg.downcast::<pyo3::types::PyBool>() {
                return Ok(Self::Bool(value.extract()?));
            } else if let Ok(value) = arg.downcast::<pyo3::types::PyInt>() {
                return Ok(Self::Int(value.extract()?));
            } else if let Ok(value) = arg.downcast::<pyo3::types::PyFloat>() {
                return Ok(Self::Float(value.extract()?));
            } else if let Ok(Some(node)) = Self::into_node_name(arg) {
                return Ok(Self::Node(node));
            } else if let Ok(dtype) = arg.extract() {
                return Ok(Self::Dtype(dtype));
            } else if let Ok(layout) = arg.extract() {
                return Ok(Self::Layout(layout));
            } else if let Ok(device) = arg.extract() {
                return Ok(Self::Device(device));
            } else if let Ok(momory_format) = arg.extract() {
                return Ok(Self::MemoryFormat(momory_format));
            } else if arg.is_none() {
                return Ok(Self::None);
            }

            Err(PyTypeError::new_err(format!(
                "failed to figure out Argument type, {arg:?}"
            )))
        }
    }

    impl Argument {
        fn into_node_name(arg: &PyAny) -> PyResult<Option<String>> {
            if arg.is_none() {
                return Ok(None);
            }
            let clsname: String = arg.getattr("__class__")?.getattr("__name__")?.extract()?;
            if clsname.as_str() == "Node" {
                Ok(Some(arg.getattr("name")?.extract()?))
            } else {
                Err(PyErr::new::<PyTypeError, _>("not an Node"))
            }
        }
    }

    /// A structure containing pertinent information
    /// about a tensor within a PyTorch program
    /// ( [reference][tensormeta] )
    ///
    /// [tensormeta]: https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/shape_prop.py#L12
    #[derive(Debug, Clone, FromPyObject)]
    pub struct TensorMeta {
        pub shape: Vec<usize>,
        pub dtype: Dtype,
        pub requires_grad: bool,
        pub stride: Vec<usize>,
        pub memory_format: Option<MemoryFormat>,
        pub is_quantized: bool,
        pub qparams: HashMap<String, PyObject>,
    }

    impl IntoPy<PyObject> for TensorMeta {
        fn into_py(self, py: Python<'_>) -> PyObject {
            let shape = self.shape.into_py(py);
            let dtype = self.dtype.into_py(py);
            let required_grad = self.requires_grad.into_py(py);
            let stride = self.stride.into_py(py);
            let memory_format = self.memory_format.into_py(py);
            let is_quantized = self.is_quantized.into_py(py);
            let qparams = self.qparams.into_py(py);

            let tensor_meta_cls = py
                .import("torch.fx.passes.shape_prop")
                .unwrap()
                .getattr("TensorMetadata")
                .unwrap();
            tensor_meta_cls
                .getattr("__new__")
                .unwrap()
                .call1((
                    tensor_meta_cls,
                    shape,
                    dtype,
                    required_grad,
                    stride,
                    memory_format,
                    is_quantized,
                    qparams,
                ))
                .unwrap()
                .into()
        }
    }

    impl TensorMeta {
        pub fn extracts_tensor_meta(tensor_meta: &PyAny) -> PyResult<Vec<Self>> {
            let clsname: String = tensor_meta
                .getattr("__class__")?
                .getattr("__name__")?
                .extract()?;
            match clsname.as_str() {
                "TensorMetadata" => Ok(vec![tensor_meta.extract()?]),
                "tuple" => tensor_meta
                    .downcast::<PyTuple>()?
                    .into_iter()
                    .map(|m| m.extract())
                    .collect::<PyResult<_>>(),
                "list" => tensor_meta
                    .downcast::<PyList>()?
                    .into_iter()
                    .map(|m| m.extract())
                    .collect::<PyResult<_>>(),
                "immutable_list" => tensor_meta
                    .downcast::<PyList>()?
                    .into_iter()
                    .map(|m| m.extract())
                    .collect::<PyResult<_>>(),
                _ => Err(PyTypeError::new_err(format!(
                    "Unknown format of tensor meta: {clsname}"
                ))),
            }
        }
    }

    /// An `enum` which represents the data type of a `torch.Tensor`.
    /// ( [reference][dtype] )
    ///
    /// [dtype]: https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum Dtype {
        Float32,
        Float64,
        Complex64,
        Complex128,
        Float16,
        Bfloat16,
        Uint8,
        Int8,
        Int16,
        Int32,
        Int64,
        Bool,
    }

    impl IntoPy<PyObject> for Dtype {
        fn into_py(self, py: Python<'_>) -> PyObject {
            let torch_module = py.import("torch").unwrap();
            match self {
                Self::Float32 => torch_module.getattr("float32").unwrap().into(),
                Self::Float64 => torch_module.getattr("float64").unwrap().into(),
                Self::Complex64 => torch_module.getattr("complex64").unwrap().into(),
                Self::Complex128 => torch_module.getattr("complex128").unwrap().into(),
                Self::Float16 => torch_module.getattr("float16").unwrap().into(),
                Self::Bfloat16 => torch_module.getattr("bfloat16").unwrap().into(),
                Self::Uint8 => torch_module.getattr("uint8").unwrap().into(),
                Self::Int8 => torch_module.getattr("int8").unwrap().into(),
                Self::Int16 => torch_module.getattr("int16").unwrap().into(),
                Self::Int32 => torch_module.getattr("int32").unwrap().into(),
                Self::Int64 => torch_module.getattr("int64").unwrap().into(),
                Self::Bool => torch_module.getattr("bool").unwrap().into(),
            }
        }
    }

    impl<'py> FromPyObject<'py> for Dtype {
        fn extract(ob: &'py PyAny) -> PyResult<Self> {
            if !check_object_type(ob, "torch", "dtype")? {
                return Err(PyTypeError::new_err(format!(
                    "Not a torch.dtype type, {ob:?}"
                )));
            }
            let py = ob.py();
            let builtins = py.import("builtins")?;
            let repr_fn = builtins.getattr("repr")?;
            let repr: String = repr_fn.call1((ob,))?.extract()?;
            match repr.as_str() {
                "torch.float32" | "torch.float" => Ok(Self::Float32),
                "torch.float64" | "torch.double" => Ok(Self::Float64),
                "torch.complex64" | "torch.cfloat" => Ok(Self::Complex64),
                "torch.complex128" | "torch.cdouble" => Ok(Self::Complex128),
                "torch.float16" | "torch.half" => Ok(Self::Float16),
                "torch.bfloat16" => Ok(Self::Bfloat16),
                "torch.uint8" => Ok(Self::Uint8),
                "torch.int8" => Ok(Self::Int8),
                "torch.int16" | "torch.short" => Ok(Self::Int16),
                "torch.int32" | "torch.int" => Ok(Self::Int32),
                "torch.int64" | "torch.long" => Ok(Self::Int64),
                "torch.bool" => Ok(Self::Bool),
                _ => Err(PyTypeError::new_err(format!(
                    "Unknown type of tensor, {ob:?}"
                ))),
            }
        }
    }

    /// An `enum` which represents the memory format
    /// on which a `torch.Tensor` is or will be allocated.
    /// ( [reference][memoryformat] )
    ///
    /// [memoryformat]: https://pytorch.org/docs/stable/tensor_attributes.html#torch-memory-format
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum MemoryFormat {
        ContiguousFormat,
        ChannelsLast,
        ChannelsLast3d,
        PreserveFormat,
    }

    impl IntoPy<PyObject> for MemoryFormat {
        fn into_py(self, py: Python<'_>) -> PyObject {
            let torch_module = py.import("torch").unwrap();
            match self {
                Self::ContiguousFormat => torch_module.getattr("contiguous_format").unwrap().into(),
                Self::ChannelsLast => torch_module.getattr("channels_last").unwrap().into(),
                Self::ChannelsLast3d => torch_module.getattr("channels_last_3d").unwrap().into(),
                Self::PreserveFormat => torch_module.getattr("preserve_format").unwrap().into(),
            }
        }
    }

    impl<'py> FromPyObject<'py> for MemoryFormat {
        fn extract(ob: &'py PyAny) -> PyResult<Self> {
            if !check_object_type(ob, "torch", "memory_format")? {
                return Err(PyTypeError::new_err(format!(
                    "Not a torch.memory_format type, {ob:?}"
                )));
            }
            let py = ob.py();
            let builtins = py.import("builtins")?;
            let repr_fn = builtins.getattr("repr")?;
            let repr: String = repr_fn.call1((ob,))?.extract()?;
            match repr.as_str() {
                "torch.contiguous_format" => Ok(Self::ContiguousFormat),
                "torch.channels_last" => Ok(Self::ChannelsLast),
                "torch.channels_last_3d" => Ok(Self::ChannelsLast3d),
                "torch.preserve_format" => Ok(Self::PreserveFormat),
                _ => Err(PyTypeError::new_err(
                    "unsupported type ob memory format".to_string(),
                )),
            }
        }
    }

    /// An `enum` which represents the memory layout of a `torch.Tensor`.
    /// ( [reference][layout] )
    ///
    /// [layout]: https://pytorch.org/docs/stable/tensor_attributes.html#torch-layout
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum Layout {
        Strided,
        SparseCoo,
    }

    impl IntoPy<PyObject> for Layout {
        fn into_py(self, py: Python<'_>) -> PyObject {
            let torch_module = py.import("torch").unwrap();
            match self {
                Self::Strided => torch_module.getattr("strided").unwrap().into(),
                Self::SparseCoo => torch_module.getattr("sparse_coo").unwrap().into(),
            }
        }
    }

    impl<'py> FromPyObject<'py> for Layout {
        fn extract(ob: &'py PyAny) -> PyResult<Self> {
            if !check_object_type(ob, "torch", "layout")? {
                return Err(PyTypeError::new_err(format!(
                    "Not a torch.layout type, {ob:?}"
                )));
            }
            let py = ob.py();
            let builtins = py.import("builtins")?;
            let repr_fn = builtins.getattr("repr")?;
            let repr: String = repr_fn.call1((ob,))?.extract()?;
            match repr.as_str() {
                "torch.strided" => Ok(Self::Strided),
                "torch.sparse_coo" => Ok(Self::SparseCoo),
                _ => Err(PyTypeError::new_err(format!(
                    "Unknown tensor layout, {ob:?}"
                ))),
            }
        }
    }

    /// An `enum` which represents the device
    /// on which a `torch.Tensor` is or will be allocated.
    /// ( [reference][device] )
    ///
    /// [device]: https://pytorch.org/docs/stable/tensor_attributes.html#torch-device
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum Device {
        Cpu(Option<usize>),
        Cuda(Option<usize>),
        Mps(Option<usize>),
    }

    impl IntoPy<PyObject> for Device {
        fn into_py(self, py: Python<'_>) -> PyObject {
            let (dev_type, index) = match self {
                Self::Cpu(index) => ("cpu", index),
                Self::Cuda(index) => ("cuda", index),
                Self::Mps(index) => ("mps", index),
            };
            let device_class = py.import("torch").unwrap().getattr("device").unwrap();
            let dev_type: PyObject = dev_type.into_py(py);
            if let Some(index) = index {
                device_class
                    .getattr("__new__")
                    .unwrap()
                    .call1((device_class, dev_type, index.into_py(py)))
                    .unwrap()
            } else {
                device_class
                    .getattr("__new__")
                    .unwrap()
                    .call1((device_class, dev_type))
                    .unwrap()
            }
            .into()
        }
    }

    impl<'py> FromPyObject<'py> for Device {
        fn extract(ob: &'py PyAny) -> PyResult<Self> {
            if !check_object_type(ob, "torch", "device")? {
                return Err(PyTypeError::new_err(format!(
                    "Not a torch.device type, {ob:?}"
                )));
            }
            let dev_type: String = ob.getattr("type")?.extract()?;
            let index = ob
                .getattr("index")
                .ok()
                .and_then(|index| index.downcast::<pyo3::types::PyInt>().ok())
                .map_or(Ok(None), |index| index.extract::<usize>().map(Some))?;
            match dev_type.as_str() {
                "cpu" => Ok(Self::Cpu(index)),
                "cuda" => Ok(Self::Cuda(index)),
                "mps" => Ok(Self::Mps(index)),
                _ => Err(PyTypeError::new_err(format!("Unknown device type, {ob:?}"))),
            }
        }
    }

    fn check_object_type(ob: &PyAny, module_name: &str, class_name: &str) -> PyResult<bool> {
        let cls = ob.getattr("__class__")?;
        Ok(
            cls.getattr("__module__")?.extract::<String>()? == module_name
                && cls.getattr("__name__")?.extract::<String>()? == class_name,
        )
    }
}

pub use custom_fn::{CustomFn, FunctionWrapper};
pub use graph::Graph;
pub use graph_module::{BufferView, GraphModule};
pub use node::Node;
pub use types::{Argument, Device, Dtype, MemoryFormat, Op, Target, TensorMeta};

#[cfg(test)]
mod tests {
    use super::types::Op;
    use pyo3::{types::PyString, IntoPy, Python};

    #[test]
    fn op_frompyobject_maps_correctly() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let s = PyString::new(py, "call_method");
            let op: Op = s.extract().unwrap();
            assert_eq!(op, Op::CallMethod);

            let s2 = PyString::new(py, "call_module");
            let op2: Op = s2.extract().unwrap();
            assert_eq!(op2, Op::CallModule);
        });
    }

    #[test]
    fn op_intopy_roundtrip_strings() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let s: String = Op::CallMethod.into_py(py).extract(py).unwrap();
            assert_eq!(s, "call_method");
            let s2: String = Op::CallModule.into_py(py).extract(py).unwrap();
            assert_eq!(s2, "call_module");
        });
    }
}
