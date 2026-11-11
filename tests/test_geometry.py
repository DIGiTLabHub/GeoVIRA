import numpy as np
import pytest
import torch
from geomed.models import GeometryAttentionEncoder as RAGE
from train_hvangle_joint_geometry_models import JointRegressionHead, predicted_relational_features, measurement_metrics
from geomed.features import hv_analytic_angles

@pytest.mark.parametrize("n_edges,dim,count", [(15,180,16768),(28,448,30592)])
def test_rage_matches_manuscript_dimensions_and_parameter_counts(n_edges,dim,count):
    model=RAGE(dim,n_edges)
    assert sum(p.numel() for p in model.parameters())==count
    edges=torch.randn(2,n_edges,4); relations=torch.randn(2,dim)
    for branch in ["full","edge_only","angle_only"]:
        y=model(relations,edges,branch=branch)
        assert y.shape==(2,64) and torch.isfinite(y).all()
    (model(relations,edges).square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

def test_coincident_predictions_have_neutral_codes_and_finite_output():
    r,e=predicted_relational_features(np.zeros((6,2),np.float32))
    assert r.shape==(180,) and e.shape==(15,4)
    np.testing.assert_array_equal(r,np.zeros(180))
    model=RAGE(180,15)
    assert torch.isfinite(model(torch.from_numpy(r)[None],torch.from_numpy(e)[None])).all()

def test_relation_translation_and_scale_invariance():
    p=np.random.default_rng(4).normal(size=(6,2)).astype(np.float32)
    r,e=predicted_relational_features(p)
    shifted,_=predicted_relational_features(p*3+np.array([5,7],np.float32))
    np.testing.assert_allclose(r,shifted,atol=3e-5)

def test_orientation_remains_sensitive_to_image_rotation():
    p=np.random.default_rng(6).normal(size=(6,2)).astype(np.float32)
    _,a=predicted_relational_features(p)
    _,b=predicted_relational_features(p@np.array([[0,1],[-1,0]],np.float32))
    np.testing.assert_allclose(a[:,:2],b[:,:2],atol=1e-5)
    assert not np.allclose(a[:,2:],b[:,2:])

def test_analytic_angles_ignore_axis_direction_and_assign_degenerate_policy():
    p=np.array([[0,0],[1,1],[3,0],[3,2],[5,0],[5,2]],float)
    a=hv_analytic_angles(p)
    np.testing.assert_allclose(a,[45,0],atol=1e-4)
    np.testing.assert_allclose(hv_analytic_angles(p[[1,0,3,2,5,4]]),a,atol=1e-5)
    assert hv_analytic_angles(np.zeros((6,2)) )==(90.,90.)

def test_joint_head_and_perfect_prediction_metrics():
    head=JointRegressionHead()
    assert head(torch.randn(4,64)).shape==(4,2)
    truth=np.arange(6,dtype=float)
    m=measurement_metrics(truth,truth,3)
    assert m["mae"]==0 and m["icc_2_1"]==pytest.approx(1)
    assert m["within_tolerance_percent"]==100 and m["bias"]==0

def test_out_of_range_or_nonfinite_landmarks_are_rejected():
    with pytest.raises(ValueError): predicted_relational_features(np.zeros((8,2)))
    p=np.zeros((6,2));p[0,0]=np.nan
    with pytest.raises(ValueError): predicted_relational_features(p)


@pytest.mark.parametrize("field", ["patient_ids", "analytic", "fold"])
def test_prediction_loader_rejects_incomplete_external_arrays(tmp_path, field):
    from types import SimpleNamespace
    from train_hvangle_joint_geometry_models import load_split
    records = [SimpleNamespace(identifier=str(i), metadata={"patient_id": i}, targets=(1., 2.)) for i in range(2)]
    arrays = dict(identifiers=np.array(["0", "1"]), patient_ids=np.array([0, 1]),
                  predicted_points=np.ones((2, 6, 2), np.float32), truth=np.array([[1., 2.], [1., 2.]]),
                  analytic=np.zeros((2, 2)), fold=np.array([0, 1]))
    arrays[field] = arrays[field][:1]
    path = tmp_path / "predictions.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="shape|patient IDs"):
        load_split(path, records, "train", 2)
