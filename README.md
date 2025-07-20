```
./build_metal.sh --build-programming-examples && ./build_Release/programming_examples/metal_example_tensor_flip
./build_metal.sh --debug --build-programming-examples && ./build_Debug/programming_examples/metal_example_tensor_flip
```

```
pytest tests/ttnn/unit_tests/operations/test_flip.py::test_flip_tiled
pytest tests/ttnn/unit_tests/operations/test_permute.py::test_permute_5d_tiled_basic
```

```
export TT_METAL_HOME=/root/tt-metal
export PYTHONPATH=$TT_METAL_HOME
export ARCH_NAME=wormhole_b0
export TT_LOGGER_TYPES=Op
export TT_LOGGER_LEVEL=Debug
export TT_METAL_DPRINT_CORES="0,0"
```
