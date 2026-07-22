import unittest

import torch
from torch import nn

from model.Model import Model
from utils.muon import HybridMuonAdamW


class HybridMuonAdamWTests(unittest.TestCase):
    def test_real_model_parameters_are_split_by_shape(self):
        model = Model(
            embedding_dim=128,
            encoder_layer_num=12,
            ff_hidden_dim=512,
            logit_clipping=50,
            demand_max1=True,
            eval_type="greedy",
        )

        optimizer = HybridMuonAdamW(model.named_parameters())

        self.assertEqual(len(optimizer.muon_param_names), 231)
        self.assertEqual(len(optimizer.adamw_param_names), 151)
        self.assertIn("position_embedding.weight", optimizer.muon_param_names)
        self.assertIn("attribute_embedding.weight", optimizer.muon_param_names)
        self.assertIn("node_type_embedding.weight", optimizer.muon_param_names)
        self.assertIn(
            "encoder.layers.0.add_n_normalization_1.norm.weight",
            optimizer.adamw_param_names,
        )

    def test_step_updates_matrix_and_vector_parameters(self):
        model = nn.Linear(3, 4, bias=True)
        optimizer = HybridMuonAdamW(model.named_parameters())
        weight_before = model.weight.detach().clone()
        bias_before = model.bias.detach().clone()

        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step()

        self.assertFalse(torch.equal(model.weight, weight_before))
        self.assertFalse(torch.equal(model.bias, bias_before))

    def test_state_dict_contains_both_optimizers(self):
        model = nn.Linear(3, 4, bias=True)
        optimizer = HybridMuonAdamW(model.named_parameters())
        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step()

        state_dict = optimizer.state_dict()

        self.assertEqual(set(state_dict), {"muon", "adamw"})

        restored_model = nn.Linear(3, 4, bias=True)
        restored_optimizer = HybridMuonAdamW(restored_model.named_parameters())
        restored_optimizer.load_state_dict(state_dict)

        self.assertEqual(len(restored_optimizer.muon.state), 1)
        self.assertEqual(len(restored_optimizer.adamw.state), 1)

    def test_rejects_parameters_with_more_than_two_dimensions(self):
        parameter = nn.Parameter(torch.ones(2, 2, 2))

        with self.assertRaisesRegex(ValueError, "only supports 2D parameters"):
            HybridMuonAdamW([("tensor", parameter)])


if __name__ == "__main__":
    unittest.main()
