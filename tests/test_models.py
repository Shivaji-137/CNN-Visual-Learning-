import numpy as np
import pytest
from cnn_scratch import CNN


def test_numpy_small_and_partial_batches_have_correct_epoch_loss():
    np.random.seed(7)
    model = CNN(conv_filters=2)
    images = np.random.rand(5, 8, 8, 1)
    labels = np.arange(5)
    model.learning_rate = 0  # Freeze weights to check the exact sample-weighted loss.
    expected = np.mean([-np.log(model.predict(x)[y]) for x, y in zip(images, labels)])
    for batch_size in (2, 32):
        history = model.train(images, labels, epochs=1, batch_size=batch_size)
        assert history['loss'][0] == pytest.approx(expected)


def test_numpy_convolution_gradient_matches_finite_difference():
    np.random.seed(9)
    model = CNN(conv_filters=2)
    image = np.random.rand(8, 8, 1)
    index = (1, 1, 0, 0)
    original = model.conv1_filters[index]
    epsilon = 1e-5
    model.conv1_filters[index] = original + epsilon
    plus = -np.log(model.predict(image)[2])
    model.conv1_filters[index] = original - epsilon
    minus = -np.log(model.predict(image)[2])
    model.conv1_filters[index] = original
    model.forward_pass(image)
    model.backward_pass(2)
    actual = (original-model.conv1_filters[index])/model.learning_rate
    assert actual == pytest.approx((plus-minus)/(2*epsilon), abs=1e-6)


def test_pooling_custom_stride_and_empty_data():
    model = CNN(conv_filters=2)
    result = model.max_pooling(np.arange(25).reshape(5, 5, 1), pool_size=3, stride=2)
    np.testing.assert_array_equal(result[:, :, 0], [[12, 14], [22, 24]])
    with pytest.raises(ValueError):
        model.train(np.empty((0, 8, 8, 1)), np.array([]))


def test_torch_logits_probabilities_multichannel_and_training_callback():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    from cnn_pytorch import TorchCNN
    model = TorchCNN(input_shape=(3, 10, 12), conv_filters=2)
    image = np.random.default_rng(1).random((10, 12, 3), dtype=np.float32)
    tensor = model._input_tensor(image)
    np.testing.assert_array_equal(tensor[0].numpy(), image.transpose(2, 0, 1))
    logits = model.model(tensor.to(model.device))
    assert not torch.allclose(logits.sum(1), torch.ones(1, device=model.device))
    probabilities = model.predict(image)
    assert probabilities.shape == (10,) and probabilities.sum() == pytest.approx(1)
    progress = []
    losses, accuracies, filters = model.train_with_visualization(
        np.stack([image]*3), np.array([0, 1, 2]), epochs=1, batch_size=2,
        progress_callback=lambda p,l,a: progress.append(p))
    assert progress == [.5, 1.0] and len(losses) == 2 and filters
    assert np.isfinite(losses).all()
    assert 0 <= model.evaluate(np.stack([image]*3), np.array([0,1,2])) <= 1


def test_tensorflow_custom_classes_and_feature_maps():
    pytest.importorskip('tensorflow')
    from cnn_tensorflow import TFCNN
    model = TFCNN(input_shape=(10, 12, 1), num_classes=3, conv_filters=2)
    image = np.ones((10, 12, 1), dtype=np.float32)
    output = model.predict(image)
    assert output.shape == (1, 3)
    assert output.sum() == pytest.approx(1)
    assert model.get_feature_maps(image).shape == (8, 10, 2)


def test_tensorflow_training_and_callback():
    pytest.importorskip('tensorflow')
    from cnn_tensorflow import TFCNN
    model = TFCNN(num_classes=3, conv_filters=2)
    images = np.random.default_rng(4).random((4, 8, 8, 1), dtype=np.float32)
    labels = np.array([0,1,2,0])
    history, _, callback = model.train_with_visualization(images,labels,epochs=1,batch_size=2)
    assert np.isfinite(history.history['loss']).all()
    assert len(callback.filters) == 1 and len(callback.batch_loss) == 2
