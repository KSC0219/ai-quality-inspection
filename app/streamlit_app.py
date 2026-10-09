"""AI Quality Inspection: demo app.

Run:  streamlit run app/streamlit_app.py
"""
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qi.config import DATA_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402
from qi.inference import Inspector  # noqa: E402

st.set_page_config(page_title="AI Quality Inspection", page_icon="🔍", layout="wide")


@st.cache_resource
def get_inspector(model_dir: str) -> Inspector:
    return Inspector(model_dir)


def sample_images(per_class: int = 3) -> list[Path]:
    """A few real test photos (if downloaded) and CAD images of each class."""
    found = []
    for cls in ("def_front", "ok_front"):
        if (DATA_DIR / "casting").is_dir():
            found += sorted((DATA_DIR / "casting").rglob(f"test/{cls}/*.jpeg"))[:per_class]
        if (DATA_DIR / "synthetic" / cls).is_dir():
            found += sorted((DATA_DIR / "synthetic" / cls).glob("*.jpeg"))[:per_class]
    return found


with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. **Inspect a part**: upload a photo of an impeller (or pick a sample) and choose a model.\n"
        "2. Read the verdict and the **heatmap**: orange shows where the model saw the defect.\n"
        "3. **CAD defect lab**: build a part from the CAD model with a defect you choose, "
        "photograph it virtually, and test the model on it.\n"
        "4. **Results**: compare Keras vs PyTorch and real vs CAD training data.")
    model_dirs = sorted(str(p) for p in MODELS_DIR.glob("*") if (p / "keras_meta.json").exists()
                        or (p / "pytorch_meta.json").exists())
    preferred = next((i for i, d in enumerate(model_dirs) if Path(d).name == "real"), 0)
    model_dir = (st.selectbox("Trained models", model_dirs, index=preferred, format_func=lambda p: Path(p).name)
                 if model_dirs else None)
    st.caption("Models are trained on the Kaggle casting dataset and/or CAD-generated images.")

tab_inspect, tab_cad, tab_results = st.tabs(["🔍 Inspect a part", "🛠️ CAD defect lab", "📊 Results"])


def show_result(img, r):
    c1, c2, c3 = st.columns(3)
    c1.image(img, caption="Input", use_container_width=True)
    if "overlay" in r:
        c2.image(r["overlay"], caption="Grad-CAM: where the model looked (orange)", use_container_width=True)
    verdict = "❌ DEFECTIVE" if r["is_defective"] else "✅ OK"
    c3.subheader(verdict)
    c3.metric("Probability of defect", f"{r['probability_defective']:.1%}")
    c3.caption(f"Model: {r['framework']} · decision threshold {r['threshold']:.2f} "
               f"(chosen so ≥98% of defects are caught on validation data)")
    if "measurement" in r:
        m = r["measurement"]
        c3.markdown(f"**Bore check vs CAD:** {m['status']}")
        if "bore_mm" in m:
            c3.caption(f"Measured {m['bore_mm']} mm · nominal {m['nominal_mm']} ± {m['tolerance_mm']} mm")
        st.image(r["measurement_overlay"], caption="Dimension check (hub used as the ruler)", width=320)


with tab_inspect:
    if not model_dir:
        st.warning("No trained model found in `models/`. Train one first: see README, step 3 "
                   "(or run the Colab notebook), then reload this page.")
    else:
        insp = get_inspector(model_dir)
        fw = st.radio("Model", insp.available, horizontal=True,
                      format_func=lambda f: {"keras": "TensorFlow/Keras", "pytorch": "PyTorch",
                                             "tflite": "TFLite (edge)"}[f])
        up = st.file_uploader("Upload a top-view photo of the part", type=["jpg", "jpeg", "png"])
        samples = sample_images()
        pick = st.selectbox("…or pick a sample image", ["-"] + [str(p) for p in samples],
                            format_func=lambda p: p if p == "-" else f"{Path(p).parent.name}/{Path(p).name}")
        measure = st.checkbox("Also check bore diameter against CAD tolerance", value=False)
        source = Image.open(up) if up else (Image.open(pick) if pick != "-" else None)
        if source is not None:
            img = np.array(source.convert("RGB"))
            with st.spinner("Inspecting…"):
                r = insp.inspect(img, fw, explain=fw != "tflite", measure=measure)
            show_result(img, r)

with tab_cad:
    st.markdown("Build an impeller **from the parametric CAD model**, add a casting defect with a "
                "boolean operation, then render a realistic photo of it. This is exactly how the "
                "synthetic training images are made.")
    try:
        from qi.cad.part import ALL_DEFECTS, ImpellerSpec, export_step, make_part, tessellate
        from qi.cad.render import render_views
        cad_ok = True
    except ImportError:
        cad_ok = False
        st.error("CadQuery is not installed: `pip install cadquery`")
    if cad_ok:
        c1, c2, c3 = st.columns(3)
        defect = c1.selectbox("Defect", ["none"] + ALL_DEFECTS)
        seed = c2.number_input("Random seed", 0, 99999, 7)
        vanes = c3.slider("Number of vanes (CAD parameter)", 5, 9, 7)
        if st.button("Build part and take photo", type="primary"):
            rng = np.random.default_rng(int(seed))
            spec = ImpellerSpec(vane_count=vanes).jittered(rng)
            with st.spinner("Building CAD solid and rendering…"):
                part, info = make_part(spec, None if defect == "none" else defect, rng)
                img = render_views(*tessellate(part), spec.outer_r, 1, rng)[0]
                with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
                    export_step(part, f.name)
                    step_bytes = Path(f.name).read_bytes()
            st.session_state["cad"] = (np.stack([img] * 3, -1), info, spec.to_dict(), step_bytes)
        if "cad" in st.session_state:
            img, info, spec_d, step_bytes = st.session_state["cad"]
            a, b = st.columns([1, 1])
            a.image(img, caption=f"Rendered photo · defect: {info['defect']}", use_container_width=True)
            b.markdown("**Ground truth from CAD**")
            b.json(info)
            b.markdown("**Part dimensions (mm)**")
            b.json({k: round(v, 2) if isinstance(v, float) else v for k, v in spec_d.items()})
            b.download_button("Download CAD model (.step)", step_bytes, file_name=f"impeller_{info['defect']}.step")
            if model_dir:
                insp = get_inspector(model_dir)
                r = insp.inspect(img, insp.available[0], explain=True, measure=defect == "bore_oversize")
                st.markdown("---\n**What the trained model says about this CAD-generated part:**")
                show_result(img, r)

with tab_results:
    comp = RESULTS_DIR / "comparison.md"
    if comp.exists():
        st.markdown(comp.read_text())
        if (RESULTS_DIR / "comparison.png").exists():
            st.image(str(RESULTS_DIR / "comparison.png"))
    else:
        st.info("No results yet. Run `python scripts/run_experiments.py` then `python scripts/compare.py`.")
    for exp_dir in sorted(p for p in RESULTS_DIR.glob("*") if p.is_dir()):
        for fw_dir in sorted(p for p in exp_dir.glob("*") if (p / "metrics.json").exists()):
            with st.expander(f"{exp_dir.name} · {fw_dir.name}"):
                m = json.loads((fw_dir / "metrics.json").read_text())
                st.json(m["test_at_chosen_threshold"])
                cols = st.columns(2)
                for i, name in enumerate(("history.png", "confusion_matrix.png")):
                    if (fw_dir / name).exists():
                        cols[i].image(str(fw_dir / name))
                if (fw_dir / "gradcam_examples.png").exists():
                    st.image(str(fw_dir / "gradcam_examples.png"))
    mc = RESULTS_DIR / "measurement_check.json"
    if mc.exists():
        st.subheader("Bore measurement accuracy (CAD dimension check)")
        st.json(json.loads(mc.read_text())["summary"])
