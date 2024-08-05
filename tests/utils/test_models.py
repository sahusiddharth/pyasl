import pytest
import numpy as np
import tensorflow as tf
from pyasl.utils.models import dilated_net_wide


class TestDilatedNetWide:

    @pytest.fixture
    def default_model(self):
        return dilated_net_wide(depth=3)

    def test_default_parameters(self, default_model):
        assert isinstance(default_model, tf.keras.Model)
        assert default_model.input_shape == (None, None, None, 1)
        assert default_model.output_shape == (None, None, None, 1)

    @pytest.mark.parametrize("depth,image_channels,filters,expansion", [
        (2, 1, 32, 4),
        (4, 3, 64, 2),
        (5, 2, 16, 8),
        (3, 4, 128, 1),
    ])
    def test_custom_parameters(self, depth, image_channels, filters, expansion):
        model = dilated_net_wide(depth=depth, image_channels=image_channels, filters=filters, expansion=expansion)
        assert model.input_shape == (None, None, None, image_channels)
        assert model.output_shape == (None, None, None, image_channels)

    def test_model_compilation(self, default_model):
        default_model.compile(optimizer='adam', loss='mse')
        assert default_model.optimizer.__class__.__name__ == 'Adam'
        assert default_model.loss == 'mse'

    @pytest.mark.parametrize("input_shape", [
        (1, 32, 32, 1),
        (1, 64, 64, 1),
        (1, 128, 128, 1),
    ])
    def test_forward_pass(self, default_model, input_shape):
        x = np.random.rand(*input_shape).astype(np.float32)
        y = default_model.predict(x)
        assert y.shape == input_shape

    def test_model_summary(self, default_model):
        summary = []
        default_model.summary(print_fn=lambda x: summary.append(x))
        summary = '\n'.join(summary)
        assert 'Total params:' in summary
        assert 'Trainable params:' in summary
        assert 'Non-trainable params: 0' in summary

# TODO: check what are the valid parameteres
    # @pytest.mark.parametrize("depth,image_channels,filters,expansion", [
    #     (0, 1, 32, 4),
    #     (-1, 1, 32, 4),
    #     (3, 0, 32, 4),
    #     (3, -1, 32, 4),
    #     (3, 1, 0, 4),
    #     (3, 1, -32, 4),    
    #     (3, 1, 32, 0),
    #     (3, 1, 32, -1),
    # ])
    # def test_invalid_parameters(self, depth, image_channels, filters, expansion):
    #     with pytest.raises(ValueError):
    #         dilated_net_wide(depth=depth, image_channels=image_channels, filters=filters, expansion=expansion)

    def test_large_depth(self):
        model = dilated_net_wide(depth=10)
        assert isinstance(model, tf.keras.Model)

    def test_large_filters(self):
        model = dilated_net_wide(depth=3, filters=1024)
        assert isinstance(model, tf.keras.Model)

    def test_large_expansion(self):
        model = dilated_net_wide(depth=3, expansion=16)
        assert isinstance(model, tf.keras.Model)

    def test_model_trainability(self, default_model):
        assert all(layer.trainable for layer in default_model.layers)

    def test_output_activation(self, default_model):
        assert default_model.layers[-1].__class__.__name__ == 'Add'

    def test_model_serialization(self, default_model):
        config = default_model.get_config()
        new_model = tf.keras.Model.from_config(config)
        assert new_model.input_shape == default_model.input_shape
        assert new_model.output_shape == default_model.output_shape

    def test_custom_input_shape(self):
        model = dilated_net_wide(depth=3, image_channels=1)
        x = np.random.rand(1, 5, 7, 1).astype(np.float32)
        y = model.predict(x)
        assert y.shape == (1, 5, 7, 1)
