import pathlib
import sys
import unittest

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "src" / "deep learning"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from train import WeightedMSELoss


class WeightedMSELossTests(unittest.TestCase):
    def test_known_loss_and_gradient_follow_target_weights(self) -> None:
        loss_fn = WeightedMSELoss((1.0, 2.0))
        pred = torch.tensor([[1.0, 1.0]], requires_grad=True)

        loss = loss_fn(pred, torch.zeros_like(pred))
        loss.backward()

        self.assertAlmostEqual(loss.item(), 1.0)
        torch.testing.assert_close(pred.grad, torch.tensor([[2.0 / 3.0, 4.0 / 3.0]]))

    def test_equal_weights_match_standard_mse(self) -> None:
        pred = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        target = torch.tensor([[0.0, 2.0], [1.0, 2.0]])

        self.assertTrue(torch.allclose(WeightedMSELoss((1.0, 1.0))(pred, target), torch.nn.functional.mse_loss(pred, target)))

    def test_scaling_all_weights_does_not_change_loss(self) -> None:
        pred = torch.tensor([[1.0, -2.0, 3.0]])
        target = torch.zeros_like(pred)

        self.assertTrue(torch.allclose(WeightedMSELoss((1.0, 2.0, 3.0))(pred, target), WeightedMSELoss((10.0, 20.0, 30.0))(pred, target)))

    def test_invalid_weights_are_rejected(self) -> None:
        for weights in (
            (),
            1.0,
            [1.0, 0.0],
            [1.0, -1.0],
            [1.0, float("nan")],
            [1.0, float("inf")],
            [3e38, 3e38],
            [[1.0, 2.0]],
        ):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    WeightedMSELoss(weights)

    def test_prediction_shape_and_width_must_match_weights(self) -> None:
        loss_fn = WeightedMSELoss((1.0, 2.0))
        with self.assertRaises(ValueError):
            loss_fn(torch.zeros(2, 2), torch.zeros(2, 1))
        with self.assertRaises(ValueError):
            loss_fn(torch.zeros(2, 3), torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            loss_fn(torch.tensor(1.0), torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
