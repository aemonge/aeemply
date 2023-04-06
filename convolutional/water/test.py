import unittest
from water import Water
import torch


class TestWater(unittest.TestCase):
    def test_water_output_size(self):
        water_layer = Water(
            image_size=(224, 256, 3), out_channels=16, kernel=(3, 3), reduction=1
        )
        input_tensor = torch.randn(1, 3, 224, 256)
        output_tensor = water_layer(input_tensor)
        self.assertEqual(output_tensor.size(), (1, 16, 224, 256))

    def test_water_get_linear_size(self):
        water_layer = Water(
            image_size=(224, 256, 3), out_channels=16, kernel=(3, 3), reduction=1
        )
        self.assertEqual(water_layer.get_linear_size(), 16 * 224 * 256)

    def test_invalid_pooling_layers(self):
        with self.assertRaises(ValueError):
            water_layer = Water(
                image_size=(224, 256, 3),
                out_channels=3,
                kernel=(2, 2),
                use_maxpool=True,
                use_avgpool=True,
            )


if __name__ == "__main__":
    unittest.main()
