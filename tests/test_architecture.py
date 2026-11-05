import numpy as np
import pytest
from architecture_builder import calculate_output_shape, generate_pytorch_code, generate_tensorflow_code


def layer(kind, **params):
    return dict(type=kind, params=params)


def test_invalid_spatial_layer_after_flatten():
    with pytest.raises(ValueError, match='spatial input'):
        calculate_output_shape([layer('Flatten'), layer('Convolutional')], (1, 8, 8))
    with pytest.raises(ValueError, match='zero size'):
        calculate_output_shape([layer('Convolutional', kernel_size=7, padding='valid')], (1, 4, 4))


@pytest.mark.parametrize('padding,kernel,stride', [('same', 2, 2), ('same', 4, 3), ('valid', 3, 2)])
def test_generated_torch_shapes_match_calculation(padding, kernel, stride):
    torch = pytest.importorskip('torch')
    layers = [layer('Convolutional', filters=2, kernel_size=kernel, stride=stride, padding=padding),
              layer('FullyConnected', units=3, activation='none')]
    namespace = {}
    exec(generate_pytorch_code(layers, (1, 10, 12)), namespace)
    actual = namespace['output'].shape[1:]
    assert list(actual) == calculate_output_shape(layers, (1, 10, 12))[-1]


def test_generated_tensorflow_dense_implicitly_flattens():
    pytest.importorskip('tensorflow')
    namespace = {}
    layers = [layer('Convolutional', filters=2, kernel_size=3, stride=1, padding='same'),
              layer('FullyConnected', units=3, activation='none')]
    exec(generate_tensorflow_code(layers, (1, 8, 8)), namespace)
    assert namespace['model'].output_shape == (None, 3)
