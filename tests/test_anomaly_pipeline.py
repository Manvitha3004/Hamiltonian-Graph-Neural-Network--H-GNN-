"""Unit tests for anomaly injection and scoring utilities."""

import pytest
import torch

from anomaly_detection import compute_anomaly_score
from anomaly_injection import inject_maneuver


class TestInjectManeuver:
    """Tests for impulsive maneuver injection."""

    def test_injects_delta_v_only_from_maneuver_timestep(self):
        positions = torch.zeros(2, 5, 3, dtype=torch.float32)
        velocities = torch.zeros(2, 5, 3, dtype=torch.float32)
        delta_v = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)

        q_new, p_new = inject_maneuver(
            positions,
            velocities,
            t_maneuver=2,
            delta_v=delta_v,
            satellite_index=1,
        )

        # Positions should not change.
        assert torch.allclose(q_new, positions)

        # Target satellite should be unchanged before maneuver and shifted after.
        assert torch.allclose(p_new[1, :2], velocities[1, :2])
        expected_after = velocities[1, 2:] + delta_v.view(1, 3)
        assert torch.allclose(p_new[1, 2:], expected_after)

        # Non-target satellite should remain unchanged.
        assert torch.allclose(p_new[0], velocities[0])

    def test_does_not_mutate_input_tensors(self):
        positions = torch.randn(2, 4, 3)
        velocities = torch.randn(2, 4, 3)
        positions_orig = positions.clone()
        velocities_orig = velocities.clone()

        q_new, p_new = inject_maneuver(
            positions,
            velocities,
            t_maneuver=1,
            delta_v=torch.tensor([0.2, 0.0, -0.1]),
            satellite_index=0,
        )

        assert torch.allclose(positions, positions_orig)
        assert torch.allclose(velocities, velocities_orig)
        assert q_new.data_ptr() != positions.data_ptr()
        assert p_new.data_ptr() != velocities.data_ptr()

    @pytest.mark.parametrize(
        "sat_idx,t_idx",
        [(-1, 1), (2, 1), (0, -1), (0, 5)],
    )
    def test_raises_for_out_of_range_indices(self, sat_idx, t_idx):
        positions = torch.zeros(2, 5, 3)
        velocities = torch.zeros(2, 5, 3)

        with pytest.raises((IndexError, ValueError)):
            inject_maneuver(
                positions,
                velocities,
                t_maneuver=t_idx,
                delta_v=torch.tensor([0.1, 0.1, 0.1]),
                satellite_index=sat_idx,
            )


class TestComputeAnomalyScore:
    """Tests for scalar anomaly score aggregation."""

    def test_matches_expected_formula_value(self, dummy_hamiltonian_model):
        model = dummy_hamiltonian_model

        # Use (N, T, 3); function should convert to time-major internally.
        q_actual = torch.zeros(2, 3, 3)
        p_actual = torch.zeros(2, 3, 3)

        q_pred = q_actual.clone()
        p_pred = p_actual.clone()

        # t=1 contributions: q error 1^2 and p error 3^2 => residual 10
        q_pred[0, 1, 0] = 1.0
        p_pred[0, 1, 1] = 3.0

        # t=2 contributions: q error 2^2 and p error 4^2 => residual 20
        q_pred[1, 2, 2] = 2.0
        p_pred[1, 2, 0] = 4.0

        score = compute_anomaly_score(
            model,
            actual_traj={"positions": q_actual, "velocities": p_actual},
            predicted_traj={"positions": q_pred, "velocities": p_pred},
        )

        # Expected:
        # max residual = 20
        # energy residuals by t from H=q.sum+2*p.sum:
        # t0:0, t1:|1 + 2*3|=7, t2:|2 + 2*4|=10 -> mean=17/3
        expected = 20.0 + 17.0 / 3.0
        assert score == pytest.approx(expected, rel=1e-6, abs=1e-6)

    def test_consistent_for_time_major_and_satellite_major_inputs(
        self,
        dummy_hamiltonian_model,
        zero_state_nt,
    ):
        model = dummy_hamiltonian_model

        q_n_t, p_n_t = zero_state_nt
        q_n_t[1, 2, 1] = 1.5
        p_n_t[0, 3, 2] = -2.5

        q_t_n = q_n_t.permute(1, 0, 2).contiguous()
        p_t_n = p_n_t.permute(1, 0, 2).contiguous()

        score_nt = compute_anomaly_score(
            model,
            actual_traj=(torch.zeros_like(q_n_t), torch.zeros_like(p_n_t)),
            predicted_traj=(q_n_t, p_n_t),
        )
        score_tn = compute_anomaly_score(
            model,
            actual_traj=(torch.zeros_like(q_t_n), torch.zeros_like(p_t_n)),
            predicted_traj=(q_t_n, p_t_n),
        )

        assert score_nt == pytest.approx(score_tn, rel=1e-6, abs=1e-6)

    def test_raises_for_mismatched_shapes(self, dummy_hamiltonian_model):
        model = dummy_hamiltonian_model

        q_actual = torch.zeros(2, 3, 3)
        p_actual = torch.zeros(2, 3, 3)
        q_pred = torch.zeros(2, 4, 3)
        p_pred = torch.zeros(2, 4, 3)

        with pytest.raises(ValueError):
            compute_anomaly_score(
                model,
                actual_traj=(q_actual, p_actual),
                predicted_traj=(q_pred, p_pred),
            )
