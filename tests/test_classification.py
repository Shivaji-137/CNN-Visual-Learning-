import numpy as np
import pytest
from PIL import Image


def test_mobilenet_preprocessing_uses_original_pixel_range():
    pytest.importorskip('tensorflow')
    from classification_playground import preprocess_image_tensorflow
    tensor, display = preprocess_image_tensorflow(Image.new('RGB', (20,20), (255,255,255)), 'MobileNetV2 (ImageNet)')
    np.testing.assert_allclose(tensor, 1.0)
    np.testing.assert_allclose(display, 1.0)


def test_digit_activation_maps_use_real_convolution_layers():
    pytest.importorskip('torch')
    pytest.importorskip('tensorflow')
    from classification_playground import (create_mnist_cnn_pytorch, create_mnist_cnn_tensorflow,
        get_class_activation_map, get_feature_visualizations, preprocess_image_pytorch, preprocess_image_tensorflow)
    image = Image.new('RGB', (28,28), (128,128,128))
    for framework, factory, preprocess in [
        ('PyTorch', create_mnist_cnn_pytorch, preprocess_image_pytorch),
        ('TensorFlow', create_mnist_cnn_tensorflow, preprocess_image_tensorflow),
    ]:
        model = factory()
        tensor, array = preprocess(image, 'MNIST CNN')
        cam, prediction = get_class_activation_map(model,tensor,array,framework,'MNIST CNN')
        assert cam.shape == (28,28) and np.isfinite(cam).all()
        features = get_feature_visualizations(model,tensor,framework,'MNIST CNN')
        assert len(features) == 2
        assert features[0]['maps'].shape[0] == 16
        assert features[1]['maps'].shape[0] == 32
