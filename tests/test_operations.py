import numpy as np
import pytest
from operations_explorer import activation_values, animation_figure, point_steps, window_steps


def test_pooling_values_and_window_positions():
    image = np.arange(1, 17).reshape(4, 4)
    maximum, steps = window_steps(image, "Max pooling")
    average, _ = window_steps(image, "Average pooling")
    np.testing.assert_array_equal(maximum, [[6, 8], [14, 16]])
    np.testing.assert_array_equal(average, [[3.5, 5.5], [11.5, 13.5]])
    assert len(steps) == 4
    assert (steps[-1]['top'], steps[-1]['left']) == (2, 2)
    assert np.isnan(steps[0]['output'][1, 1])
    np.testing.assert_array_equal(steps[-1]['output'], maximum)


def test_convolution_is_cnn_cross_correlation_and_skips_partial_windows():
    image = np.arange(25).reshape(5, 5)
    kernel = np.array([[1, 2], [3, 4]])
    output, steps = window_steps(image, "Convolution", 2, 2, kernel)
    assert output.shape == (2, 2)
    assert output[0, 0] == (image[:2, :2]*kernel).sum()
    assert len(steps) == 4


@pytest.mark.parametrize("name", ["ReLU", "Sigmoid", "Tanh"])
def test_activation_animation_final_frame(name):
    image = np.array([[-1000, -1], [0, 1000]])
    with np.errstate(over='raise', invalid='raise'):
        output, steps = point_steps(image, "Activation", name)
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output, activation_values(image, name))
    np.testing.assert_array_equal(steps[-1]['output'], output)


def test_flatten_preserves_row_order_and_export_includes_playback():
    image = np.array([[1, 3], [2, 4]])
    output, steps = point_steps(image, "Flatten")
    np.testing.assert_array_equal(output, [[1, 3, 2, 4]])
    fig = animation_figure(image, output, steps, "Flatten")
    assert len(fig.frames) == 4
    html = fig.to_html(include_plotlyjs=True)
    assert 'Plotly.addFrames' in html and 'Restart' in html


@pytest.mark.parametrize('size,stride', [(0, 2), (5, 1), (2, 0)])
def test_invalid_windows(size, stride):
    with pytest.raises(ValueError):
        window_steps(np.ones((4, 4)), 'Max pooling', size, stride)


def test_sample_pipeline_convolution_matches_animation_and_leaky_relu():
    from complete_cnn_copy import apply_convolution, apply_activation
    image = np.arange(36,dtype=float).reshape(6,6)
    kernel=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    output,_=window_steps(image,'Convolution',3,1,kernel)
    np.testing.assert_array_equal(output,apply_convolution(image,kernel))
    values=np.array([[-2,-1],[0,2]],dtype=float)
    np.testing.assert_array_equal(activation_values(values,'Leaky ReLU'),apply_activation(values,'Leaky ReLU'))
