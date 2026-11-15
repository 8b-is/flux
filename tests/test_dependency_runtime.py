import unittest

import cv2
import numpy as np
import torch

from flux.model import Flux, FluxParams


class DependencyRuntimeTest(unittest.TestCase):
    def test_flux_cpu_forward(self):
        parameters = FluxParams(
            in_channels=4, out_channels=4, vec_in_dim=4, context_in_dim=4,
            hidden_size=24, mlp_ratio=2, num_heads=2, depth=1,
            depth_single_blocks=1, axes_dim=[4, 4, 4], theta=10000,
            qkv_bias=True, guidance_embed=False,
        )
        model = Flux(parameters).eval()
        with torch.no_grad():
            output = model(
                torch.zeros(1, 4, 4), torch.zeros(1, 4, 3),
                torch.zeros(1, 2, 4), torch.zeros(1, 2, 3),
                torch.ones(1), torch.zeros(1, 4),
            )
        self.assertEqual(tuple(output.shape), (1, 4, 4))
        self.assertTrue(torch.isfinite(output).all())

    def test_canny_with_numpy_image(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[4:12, 4:12] = 255
        edges = cv2.Canny(image, 50, 200)
        self.assertEqual(edges.shape, (16, 16))
        self.assertGreater(np.count_nonzero(edges), 0)
