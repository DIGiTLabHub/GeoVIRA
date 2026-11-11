"""Exercise final interfaces without encoders, images or detector checkpoints."""
import sys
import numpy as np
import pytest
import torch
from train_predicted_landmark_frozen_vision_fusion import GAEFrozenVisionFusion, patient_derangement, sattolo_derangement
from train_predicted_landmark_mii_cross_attention import MIICrossAttentionGAE
from train_hvangle_axis_aligned_mii import AxisAlignedGAE
from train_aasce_cobb_measurement_line_mii import CobbLineAlignedGAE

@pytest.mark.parametrize('task,dim,edges,out', [('foot',180,15,2),('cobb',448,28,1)])
def test_interfaces_forward_backward(task,dim,edges,out):
    torch.manual_seed(17)
    relational=torch.randn(2,dim); pairwise=torch.randn(2,edges,4)
    models=[(GAEFrozenVisionFusion(dim,edges,out,image_dim=1024),(torch.randn(2,1024),)),
            (MIICrossAttentionGAE(dim,edges,out),(torch.randn(2,225,2048),))]
    if task=='foot': models.append((AxisAlignedGAE(dim,edges),(torch.randn(2,3,2048),'axis_aligned_matched')))
    else: models.append((CobbLineAlignedGAE(dim,edges),(torch.randn(2,2,2048),)))
    for model,args in models:
        prediction=model(relational,pairwise,*args)
        assert prediction.shape==(2,out) and torch.isfinite(prediction).all()
        prediction.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

def test_case_donors_are_deterministic_and_exclude_same_patient():
    patients=np.repeat(np.arange(5),2)
    p=patient_derangement(patients,42)
    np.testing.assert_array_equal(p,patient_derangement(patients,42))
    assert sorted(p)==list(range(10))
    assert np.all(patients[p]!=patients)
    q=sattolo_derangement(20,17)
    assert sorted(q)==list(range(20)) and np.all(q!=np.arange(20))


def test_paper_interface_defaults(monkeypatch):
    import train_predicted_landmark_frozen_vision_fusion as global_interface
    import train_localized_mii_image_shuffled as aligned_control
    import audit_ordinary_attention_geometry_alignment as attention_audit
    monkeypatch.setattr(sys, "argv", ["global"])
    args = global_interface.parse_args()
    assert args.encoder == "medimageinsight" and args.device == "auto"
    monkeypatch.setattr(sys, "argv", ["aligned", "--task", "foot"])
    args = aligned_control.parse_args()
    assert args.shuffle_mode == "feature-map" and args.device == "auto"
    monkeypatch.setattr(sys, "argv", ["audit"])
    assert attention_audit.parse_args().device == "auto"


def test_baseline_outputs_feed_ablation_inputs():
    import train_paired_joint_geometry_models as foot_geometry
    import train_joint_geometry_models as cobb_geometry
    import run_rage_feature_ablation as ablation
    assert foot_geometry.DEFAULT_OUTPUT / "results.json" == ablation.TASKS["foot"]["full_results"]
    assert cobb_geometry.DEFAULT_OUTPUT / "results.json" == ablation.TASKS["cobb"]["full_results"]
