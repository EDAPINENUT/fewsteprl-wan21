#!/usr/bin/env bash
# Make all 16 VBench dimensions importable in a dedicated scoring environment.
#
# Use a copy of the training env (e.g. `conda create -n vbench-scoring --clone
# fewsteprl-wan21`), activate it, then:
#
#   bash scripts/eval/setup_vbench_env.sh            # apply the fixes, then verify
#   bash scripts/eval/setup_vbench_env.sh --verify   # verify only
#
# Why a separate env: `vbench` is installed with --no-deps in the training env
# (its metadata would pull transformers back to 4.33 and break the Wan model
# path), which leaves 12 of the 16 scorers unimportable. The fixes below are
# scoped to the scoring env; the training env stays stock.
#
#   1. detectron2 (object_class / multiple_objects / color / spatial_relationship):
#      not on PyPI, built from source against the env's torch.
#   2. `from pkg_resources import packaging` in clip and ViCLIP (setuptools>=70
#      dropped it) -> `import packaging.version`.
#   3. tag2text (scene) imports apply_chunking_to_forward & co. from
#      transformers.modeling_utils; they live in transformers.pytorch_utils.
#   4. torch>=2.6 defaults torch.load(weights_only=True); every VBench checkpoint
#      is a full pickle. A sitecustomize shim restores the old default in this
#      env only (only the default changes; an explicit weights_only= wins).
#   5. CPU fallback for torchvision.ops.nms, for torchvision builds without CUDA
#      NMS kernels (numerically identical: NMS is index selection).

set -uo pipefail

PY="${PYTHON:-python3}"
SP="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

verify() {
    VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-models/vbench}" "${PY}" - <<'PYEOF'
import importlib, sys
DIMS = ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality",
        "object_class", "multiple_objects", "color", "spatial_relationship",
        "scene", "temporal_style", "overall_consistency", "human_action",
        "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style"]
bad = []
for d in DIMS:
    try:
        getattr(importlib.import_module(f"vbench.{d}"), f"compute_{d}")
    except Exception as e:
        bad.append(f"{d}: {type(e).__name__}: {e}")
import torch, transformers, numpy
print(f"  torch={torch.__version__} transformers={transformers.__version__} numpy={numpy.__version__}")
print(f"  {16 - len(bad)}/16 dimensions importable")
for b in bad:
    print(f"  FAIL {b}")
sys.exit(1 if bad else 0)
PYEOF
}

if [[ "${1:-}" == "--verify" ]]; then
    verify; exit $?
fi

fail=0

# 1. detectron2
if ! "${PY}" -c "import detectron2" >/dev/null 2>&1; then
    echo "--- building detectron2 from source ---"
    "${PY}" -m pip install --no-build-isolation --no-input -q \
        'git+https://github.com/facebookresearch/detectron2.git' || { echo "FAILED: detectron2"; fail=1; }
fi

# 2. pkg_resources.packaging
for f in "${SP}/clip/clip.py" "${SP}/vbench/third_party/ViCLIP/viclip_text.py"; do
    [[ -f "${f}" ]] && sed -i 's/^from pkg_resources import packaging$/import packaging.version/' "${f}"
done

# 3. tag2text: helpers moved to transformers.pytorch_utils
MED="${SP}/vbench/third_party/tag2Text/med.py"
if [[ -f "${MED}" ]]; then
    "${PY}" - "${MED}" <<'PYEOF'
import sys
f = sys.argv[1]
s = open(f).read()
old = """from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)"""
new = """from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)"""
if old in s:
    open(f, "w").write(s.replace(old, new))
PYEOF
fi

# 4 + 5. sitecustomize shims
"${PY}" - "${SP}/sitecustomize.py" <<'PYEOF'
import os, sys
p = sys.argv[1]
s = open(p).read() if os.path.exists(p) else ""
if "_vbench_weights_only_shim" not in s:
    s += '''
try:
    import torch as _torch
    if not getattr(_torch, "_vbench_weights_only_shim", False):
        _orig_load = _torch.load
        def _load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)
        _torch.load = _load
        _torch._vbench_weights_only_shim = True
except Exception:
    pass
'''
if "_vbench_cpu_nms_shim" not in s:
    s += '''
try:
    import torchvision as _tv
    if not getattr(_tv, "_vbench_cpu_nms_shim", False):
        _orig_nms = _tv.ops.nms
        def _nms_cpu(boxes, scores, iou_threshold):
            if boxes.is_cuda:
                return _orig_nms(boxes.detach().cpu(), scores.detach().cpu(), iou_threshold).to(boxes.device)
            return _orig_nms(boxes, scores, iou_threshold)
        _tv.ops.nms = _nms_cpu
        _tv.ops.boxes.nms = _nms_cpu
        _tv._vbench_cpu_nms_shim = True
except Exception:
    pass
'''
open(p, "w").write(s)
PYEOF

echo "--- verification ---"
verify || fail=1
exit "${fail}"
