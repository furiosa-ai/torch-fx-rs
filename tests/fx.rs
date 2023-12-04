use std::{collections::HashMap, sync::Arc};

use pyo3::{
    types::{PyDict, PyList, PyModule, PySlice, PyTuple},
    IntoPy, Py, PyResult, Python,
};

use torch_fx_rs::{
    Argument, CustomFn, Device, Dtype, FunctionWrapper, Graph, GraphModule, MemoryFormat, Op,
    Target, TensorMeta,
};

#[test]
fn unittest_graph() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let graph = Graph::new(py)?;
        graph.create_node(
            Op::GetAttr,
            Target::Str("test".into()),
            vec![],
            Default::default(),
            "test_node",
            Default::default(),
        )?;
        let nodes = graph.nodes()?;
        assert_eq!(nodes.count(), 1);
        let po: Py<Graph> = graph.into();
        let graph: &Graph = po.as_ref(py);
        let nodes = graph.nodes()?;
        assert_eq!(nodes.count(), 1);
        Ok(())
    })
}

#[test]
fn unittest_init_methods() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let graph = Graph::new(py)?;
        let gm = GraphModule::new_with_empty_gm(py, graph)?;
        let graph = Graph::new(py)?;
        GraphModule::new(py, gm, graph)?;
        Ok(())
    })
}

#[test]
fn unittest_copy_graph_python() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let gm = PyModule::from_code(
            py,
            r#"
import torch.fx
import torch.nn
import operator

graph = torch.fx.Graph()
i0 = graph.placeholder("i0")
i1 = graph.placeholder("i1")
g0 = graph.create_node("call_function", operator.getitem, (i0,0), name="getitem_0")
g1 = graph.create_node("call_function", operator.getitem, (i1,0), name="getitem_1")
graph.output((g0, g1))
gm = torch.fx.GraphModule(torch.nn.Module(), graph)
                "#,
            "copy_graph.py",
            "copy_graph",
        )?;
        let gm: &GraphModule = gm.getattr("gm")?.downcast()?;
        let mut mapper = HashMap::default();
        let graph = Graph::new(py)?;
        graph.new_placeholder("i_alt0")?;
        graph.new_placeholder("i_alt1")?;
        let g0 = gm.graph()?.lookup_node("getitem_0")?.unwrap();
        mapper.insert(
            gm.graph()?.flatten_node_args(g0.name()?)?.unwrap()[0].clone(),
            "i_alt0".to_string(),
        );
        graph.copy_node(g0, Some(&mapper))?;
        let g1 = gm.graph()?.lookup_node("getitem_1")?.unwrap();
        mapper.insert(
            gm.graph()?.flatten_node_args(g1.name()?)?.unwrap()[0].clone(),
            "i_alt1".to_string(),
        );
        graph.copy_node(g1, Some(&mapper))?;
        let output_arg = Argument::NodeTuple(vec![g0.name()?.clone(), g1.name()?.clone()]);
        graph.new_output(output_arg)?;
        graph.eliminate_dead_code()?;
        graph.lint()?;
        assert_eq!(graph.named_nodes()?.len(), 5);
        let gm = GraphModule::new(py, gm, graph)?;
        assert_eq!(gm.graph()?.flatten_node_args("output")?.unwrap().len(), 2);
        Ok(())
    })
}

#[test]
fn unittest_copy_graph_rust() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let operator = py.import("operator")?;
        let getitem = operator.getattr("getitem")?;

        let graph_from = Graph::new(py)?;
        let i0 = graph_from.new_placeholder("i0")?;
        let i1 = graph_from.new_placeholder("i1")?;
        let _g0 = graph_from.create_node(
            Op::CallFunction,
            Target::BuiltinFn("getitem".to_string(), getitem.into_py(py)),
            vec![Argument::Value(i0.into_py(py)), Argument::Int(0)],
            Default::default(),
            "getitem_0",
            Default::default(),
        )?;
        let _g1 = graph_from.create_node(
            Op::CallFunction,
            Target::BuiltinFn("getitem".to_string(), getitem.into_py(py)),
            vec![Argument::Value(i1.into_py(py)), Argument::Int(0)],
            Default::default(),
            "getitem_1",
            Default::default(),
        )?;
        graph_from.new_output(Argument::NodeTuple(vec![
            "getitem_0".to_string(),
            "getitem_1".to_string(),
        ]))?;

        let gm = GraphModule::new_with_empty_gm(py, graph_from)?;

        let mut mapper = HashMap::default();
        let graph_to = Graph::new(py)?;
        graph_to.new_placeholder("i_alt0")?;
        graph_to.new_placeholder("i_alt1")?;
        let g0 = gm.graph()?.lookup_node("getitem_0")?.unwrap();
        mapper.insert(
            gm.graph()?.flatten_node_args(g0.name()?)?.unwrap()[0].clone(),
            "i_alt0".to_string(),
        );
        graph_to.copy_node(g0, Some(&mapper))?;
        let g1 = gm.graph()?.lookup_node("getitem_1")?.unwrap();
        mapper.insert(
            gm.graph()?.flatten_node_args(g1.name()?)?.unwrap()[0].clone(),
            "i_alt1".to_string(),
        );
        graph_to.copy_node(g1, Some(&mapper))?;
        let output_arg = Argument::NodeTuple(vec![g0.name()?.clone(), g1.name()?.clone()]);
        graph_to.new_output(output_arg)?;
        graph_to.eliminate_dead_code()?;
        graph_to.lint()?;
        assert_eq!(graph_to.named_nodes()?.len(), 5);
        let gm = GraphModule::new(py, gm, graph_to)?;
        assert_eq!(gm.graph()?.flatten_node_args("output")?.unwrap().len(), 2);
        Ok(())
    })
}

fn generate_empty_fn() -> FunctionWrapper {
    Arc::new(|args, _kwargs| Ok(42usize.into_py(args.py())))
}

#[test]
fn unittest_custom_fn() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let g = Graph::new(py)?;

        let custom_fn = CustomFn::new("empty_fn", generate_empty_fn());
        let custom_fn_in_py = custom_fn.clone().into_py(py);
        let callable_fn = py.import("builtins")?.getattr("callable")?;
        assert!(callable_fn.call1((custom_fn_in_py,))?.extract()?);

        g.new_custom_fn("test", vec![], None, custom_fn)?;
        Ok(())
    })
}

#[test]
fn unittest_users_and_flatten_node_args() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let g = Graph::new(py)?;

        let custom_fn = CustomFn::new("empty_fn", generate_empty_fn());
        let n1_name = g
            .new_custom_fn("test", vec![], None, custom_fn.clone())?
            .name()?
            .clone();
        let n2_name = g
            .new_custom_fn(
                "test_2",
                vec![Argument::Node(n1_name.clone())],
                None,
                custom_fn,
            )?
            .name()?
            .clone();
        assert_eq!(g.users(n1_name)?.unwrap().len(), 1);
        assert_eq!(g.flatten_node_args(n2_name)?.unwrap().len(), 1);
        Ok(())
    })
}

#[test]
fn unittest_extract_buffers_python() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let gm = PyModule::from_code(
            py,
            r#"
import torch
import torch.fx
import torch.nn
import operator

graph = torch.fx.Graph()
i0 = graph.placeholder("i0")
g0 = graph.create_node("call_function", operator.getitem, (i0,0), name="getitem_0")
graph.output((g0,))
gm = torch.fx.GraphModule(torch.nn.Module(), graph)
gm._buffers["buf_buf"] = torch.randn(1)
gm._parameters["param_buf"] = torch.randn(8)
                "#,
            "extract_buffers.py",
            "extract_buffers",
        )?;
        let gm: &GraphModule = gm.getattr("gm")?.downcast()?;

        assert_eq!(gm.extract_buffers()?.len(), 1);
        assert!(gm.extract_buffers()?.get("buf_buf").is_some());
        assert_eq!(gm.extract_buffers()?.get("buf_buf").unwrap().len(), 4);
        assert_eq!(gm.count_parameters()?, 1);
        assert!(gm.get_parameter("param_buf")?.is_some());
        assert_eq!(gm.get_parameter("param_buf")?.unwrap().len(), 8 * 4);
        Ok(())
    })
}

#[test]
fn unittest_extract_buffers_rust() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let gm = GraphModule::new_with_empty_gm(py, {
            let graph = Graph::new(py)?;
            let i0 = graph.new_placeholder("i0")?;
            let _g0 = graph.create_node(
                Op::CallFunction,
                Target::BuiltinFn(
                    "getitem".to_string(),
                    py.import("operator")?.getattr("getitem")?.into_py(py),
                ),
                vec![Argument::Value(i0.into_py(py)), Argument::Int(0)],
                Default::default(),
                "getitem_0",
                Default::default(),
            )?;
            graph.new_output(Argument::NodeTuple(vec!["getitem_0".to_string()]))?;
            graph
        })?;

        let randn = py.import("torch")?.getattr("randn")?;
        gm.getattr("_buffers")?
            .set_item("buf_buf", randn.call1((1,))?)?;
        gm.getattr("_parameters")?
            .set_item("param_buf", randn.call1((8,))?)?;

        assert_eq!(gm.extract_buffers()?.len(), 1);
        assert!(gm.extract_buffers()?.get("buf_buf").is_some());
        assert_eq!(gm.extract_buffers()?.get("buf_buf").unwrap().len(), 4);
        assert_eq!(gm.count_parameters()?, 1);
        assert!(gm.get_parameter("param_buf")?.is_some());
        assert_eq!(gm.get_parameter("param_buf")?.unwrap().len(), 8 * 4);
        Ok(())
    })
}

#[test]
fn unittest_extract_strided_buffers_python() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let gm = PyModule::from_code(
            py,
            r#"
import torch
import torch.fx
import torch.nn
import operator

graph = torch.fx.Graph()
i0 = graph.placeholder("i0")
g0 = graph.create_node("call_function", operator.getitem, (i0,0), name="getitem_0")
graph.output((g0,))
gm = torch.fx.GraphModule(torch.nn.Module(), graph)
a = torch.randn(4, 3)
gm._buffers["buf_buf"] = a[1:3, 0:2]
gm._parameters["param_buf"] = a.permute([1, 0])
    "#,
            "extract_strided_buffers.py",
            "extract_strided_buffers",
        )?;
        let gm: &GraphModule = gm.getattr("gm")?.downcast()?;

        assert_eq!(gm.count_buffers()?, 1);
        assert!(gm.get_buffer("buf_buf")?.is_some());
        assert_eq!(gm.get_buffer("buf_buf")?.unwrap().len(), 9 * 4);
        assert_eq!(gm.count_parameters()?, 1);
        assert!(gm.get_parameter("param_buf")?.is_some());
        assert_eq!(gm.get_parameter("param_buf")?.unwrap().len(), 12 * 4);
        Ok(())
    })
}

#[test]
fn unittest_extract_strided_buffers_rust() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let gm = GraphModule::new_with_empty_gm(py, {
            let graph = Graph::new(py)?;
            let i0 = graph.new_placeholder("i0")?;
            let _g0 = graph.create_node(
                Op::CallFunction,
                Target::BuiltinFn(
                    "getitem".to_string(),
                    py.import("operator")?.getattr("getitem")?.into_py(py),
                ),
                vec![Argument::Value(i0.into_py(py)), Argument::Int(0)],
                Default::default(),
                "getitem_0",
                Default::default(),
            )?;
            graph.new_output(Argument::NodeTuple(vec!["getitem_0".to_string()]))?;
            graph
        })?;

        let randn = py.import("torch")?.getattr("randn")?;
        let a = randn.call1((4, 3))?;
        gm.getattr("_buffers")?.set_item(
            "buf_buf",
            a.get_item(PyTuple::new(
                py,
                vec![PySlice::new(py, 1, 3, 1), PySlice::new(py, 0, 2, 1)],
            ))?,
        )?;
        gm.getattr("_parameters")?.set_item(
            "param_buf",
            a.getattr("permute")?
                .call1((PyList::new(py, vec![1, 0]),))?,
        )?;

        assert_eq!(gm.count_buffers()?, 1);
        assert!(gm.get_buffer("buf_buf")?.is_some());
        assert_eq!(gm.get_buffer("buf_buf")?.unwrap().len(), 9 * 4);
        assert_eq!(gm.count_parameters()?, 1);
        assert!(gm.get_parameter("param_buf")?.is_some());
        assert_eq!(gm.get_parameter("param_buf")?.unwrap().len(), 12 * 4);
        Ok(())
    })
}

#[test]
fn unittest_extract_tensor_meta() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let gm = PyModule::from_code(
            py,
            r#"
import torch
from torch.fx.passes.shape_prop import TensorMetadata

tensor_meta = TensorMetadata(
    shape=torch.Size((1,2,3)),
    dtype=torch.float32,
    requires_grad=False,
    stride=(1,2,3),
    memory_format=torch.preserve_format,
    is_quantized=False,
    qparams={'123':123},
)
tensor_meta_lst = [TensorMetadata(
    shape=torch.Size((1,2,3)),
    dtype=torch.float32,
    requires_grad=False,
    stride=(1,2,3),
    memory_format=torch.preserve_format,
    is_quantized=False,
    qparams={'123':123},
), TensorMetadata(
    shape=torch.Size((1,2,3)),
    dtype=torch.float64,
    requires_grad=False,
    stride=(1,2,3),
    memory_format=torch.preserve_format,
    is_quantized=False,
    qparams={'123':123},
)]
tensor_meta_tup = (TensorMetadata(shape=torch.Size([1, 1000]),
    dtype=torch.float32,
    requires_grad=True,
    stride=(1000, 1),
    memory_format=torch.contiguous_format,
    is_quantized=False,
    qparams={}),)
    "#,
            "tensor_meta.py",
            "tensor_meta",
        )?;
        let tensor_meta = gm.getattr("tensor_meta")?;
        let tensor_meta = TensorMeta::extracts_tensor_meta(tensor_meta)?;
        assert_eq!(tensor_meta.len(), 1);
        assert_eq!(tensor_meta[0].shape, [1, 2, 3]);
        assert_eq!(tensor_meta[0].dtype, Dtype::Float32);
        assert!(!tensor_meta[0].requires_grad);
        assert_eq!(tensor_meta[0].stride, [1, 2, 3]);
        assert_eq!(
            tensor_meta[0].memory_format,
            Some(MemoryFormat::PreserveFormat)
        );
        assert!(!tensor_meta[0].is_quantized);
        assert_eq!(tensor_meta[0].qparams.len(), 1);

        let tensor_meta_lst = gm.getattr("tensor_meta_lst")?;
        let tensor_meta_lst = TensorMeta::extracts_tensor_meta(tensor_meta_lst)?;
        assert_eq!(tensor_meta_lst.len(), 2);
        assert_eq!(tensor_meta_lst[0].dtype, Dtype::Float32);
        assert_eq!(tensor_meta_lst[1].dtype, Dtype::Float64);

        let tensor_meta_tup = gm.getattr("tensor_meta_tup")?;
        let tensor_meta_tup = TensorMeta::extracts_tensor_meta(tensor_meta_tup)?;
        assert_eq!(tensor_meta_tup.len(), 1);
        assert_eq!(tensor_meta_tup[0].dtype, Dtype::Float32);
        Ok(())
    })
}

#[test]
fn unittest_into_py_tensor_meta() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let tensor_meta = TensorMeta {
            shape: vec![1, 2, 3],
            dtype: Dtype::Uint8,
            requires_grad: false,
            stride: vec![2],
            memory_format: Some(MemoryFormat::ChannelsLast),
            is_quantized: false,
            qparams: Default::default(),
        };
        let tensor_meta_py = tensor_meta.clone().into_py(py);
        let extracted_tensor_meta: TensorMeta = tensor_meta_py.extract(py)?;
        assert_eq!(tensor_meta.shape, extracted_tensor_meta.shape);
        assert_eq!(tensor_meta.dtype, extracted_tensor_meta.dtype);
        assert_eq!(
            tensor_meta.requires_grad,
            extracted_tensor_meta.requires_grad
        );
        assert_eq!(tensor_meta.stride, extracted_tensor_meta.stride);
        assert_eq!(
            tensor_meta.memory_format,
            extracted_tensor_meta.memory_format
        );
        assert_eq!(tensor_meta.is_quantized, extracted_tensor_meta.is_quantized);
        Ok(())
    })
}

#[test]
fn unittest_extract_device() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let torch_device = PyModule::from_code(
            py,
            r#"
import torch

device1 = torch.device('cpu', 0)
device2 = torch.device('cpu')
    "#,
            "torch_device.py",
            "torch_device",
        )?;

        let device1 = torch_device.getattr("device1")?;
        let device1: Device = device1.extract()?;
        assert!(matches!(device1, Device::Cpu(Some(0))));

        let device2 = torch_device.getattr("device2")?;
        let device2: Device = device2.extract()?;
        assert!(matches!(device2, Device::Cpu(None)));
        Ok(())
    })
}

#[test]
fn unittest_into_py_device() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let device1 = Device::Cpu(None);
        let torch_device_py1 = device1.into_py(py);
        let extracted_device1: Device = torch_device_py1.extract(py)?;
        assert_eq!(device1, extracted_device1);

        let device2 = Device::Cuda(Some(0));
        let torch_device_py2 = device2.into_py(py);
        let extracted_device2: Device = torch_device_py2.extract(py)?;
        assert_eq!(device2, extracted_device2);

        assert_ne!(extracted_device1, extracted_device2);
        Ok(())
    })
}

#[test]
fn unittest_extract_tensor_as_slices_python() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let gm = PyModule::from_code(
            py,
            r#"
import torch
import torch.fx
import torch.nn
import operator

graph = torch.fx.Graph()
i0 = graph.placeholder("i0")
g0 = graph.create_node("call_function", operator.getitem, (i0,0), name="getitem_0")
graph.output((g0,))
gm = torch.fx.GraphModule(torch.nn.Module(), graph)
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
z = x[:, 1:2]
gm._parameters["param_buf"] = x.t()
gm._parameters["param_slice"] = z.t()
gm._buffers["buf_buf"] = x
gm._buffers["buf_slice"] = z
    "#,
            "extract_slice.py",
            "extract_slice",
        )?;
        let gm: &GraphModule = gm.getattr("gm")?.downcast()?;

        assert_eq!(gm.count_parameters()?, 2);
        assert_eq!(
            gm.get_parameter("param_buf")?,
            Some(
                [1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0]
                    .as_slice()
            )
        );
        assert_eq!(
            gm.get_parameter("param_slice")?,
            Some([2u8, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0].as_slice())
        );

        assert_eq!(gm.count_buffers()?, 2);
        assert_eq!(
            gm.get_buffer("buf_buf")?,
            Some(
                [1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0]
                    .as_slice()
            )
        );
        assert_eq!(
            gm.get_buffer("buf_slice")?,
            Some([2u8, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0].as_slice())
        );
        Ok(())
    })
}

#[test]
fn unittest_extract_tensor_as_slices_rust() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let gm = GraphModule::new_with_empty_gm(py, {
            let graph = Graph::new(py)?;
            let i0 = graph.new_placeholder("i0")?;
            let _g0 = graph.create_node(
                Op::CallFunction,
                Target::BuiltinFn(
                    "getitem".to_string(),
                    py.import("operator")?.getattr("getitem")?.into_py(py),
                ),
                vec![Argument::Value(i0.into_py(py)), Argument::Int(0)],
                Default::default(),
                "getitem_0",
                Default::default(),
            )?;
            graph.new_output(Argument::NodeTuple(vec!["getitem_0".to_string()]))?;
            graph
        })?;

        let torch = py.import("torch")?;
        let x = torch.getattr("tensor")?.call(
            (PyList::new(
                py,
                vec![
                    PyList::new(py, vec![1, 2, 3]),
                    PyList::new(py, vec![4, 5, 6]),
                ],
            ),),
            Some({
                let kwargs = PyDict::new(py);
                kwargs.set_item("dtype", torch.getattr("int32")?)?;
                kwargs
            }),
        )?;
        let z = x.get_item(PyTuple::new(
            py,
            vec![PySlice::full(py), PySlice::new(py, 1, 2, 1)],
        ))?;

        let parameters = gm.getattr("_parameters")?;
        parameters.set_item("param_buf", x.getattr("t")?.call0()?)?;
        parameters.set_item("param_slice", z.getattr("t")?.call0()?)?;
        let buffers = gm.getattr("_buffers")?;
        buffers.set_item("buf_buf", x)?;
        buffers.set_item("buf_slice", z)?;

        assert_eq!(gm.count_parameters()?, 2);
        assert_eq!(
            gm.get_parameter("param_buf")?,
            Some(
                [1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0]
                    .as_slice()
            )
        );
        assert_eq!(
            gm.get_parameter("param_slice")?,
            Some([2u8, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0].as_slice())
        );

        assert_eq!(gm.count_buffers()?, 2);
        assert_eq!(
            gm.get_buffer("buf_buf")?,
            Some(
                [1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0]
                    .as_slice()
            )
        );
        assert_eq!(
            gm.get_buffer("buf_slice")?,
            Some([2u8, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0].as_slice())
        );
        Ok(())
    })
}
