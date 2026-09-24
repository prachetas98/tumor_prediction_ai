import os
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import pydicom
from collections import defaultdict
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import joblib

# ======================
# CONFIG / PATHS
# ======================
PROJECT_ROOT = Path("C:/Users/caree/OneDrive/Documents/capstone/tumor_pred_ai/src")
MODEL_DIR    = PROJECT_ROOT / "models" / "final"
FULL_MODEL_PATH = MODEL_DIR / "multimodal_lora_full_model.h5"

CLIP_HU = (-200, 300)
TARGET_SHAPE = (96, 96, 48)   # must match training

# ======================
# LoRA LAYER (same as training)
# ======================
class LoRADense(layers.Layer):
    def __init__(
        self,
        units,
        rank=None,
        r=None,
        alpha=16.0,
        dropout=0.0,
        activation=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if rank is None and r is not None:
            rank = r

        self.units = units
        self.rank = int(rank) if rank is not None else 0
        self.alpha = alpha
        self.dropout_rate = dropout
        self.activation = tf.keras.activations.get(activation)
        self.scaling = (alpha / self.rank) if self.rank and self.rank > 0 else 1.0

        self.base_dense = layers.Dense(units, use_bias=True)
        self.dropout = layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        if self.rank and self.rank > 0:
            self.lora_A = self.add_weight(
                name="lora_A",
                shape=(in_dim, self.rank),
                initializer="zeros",
                trainable=True,
            )
            self.lora_B = self.add_weight(
                name="lora_B",
                shape=(self.rank, self.units),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.lora_A = None
            self.lora_B = None
        super().build(input_shape)

    def call(self, inputs, training=None):
        base_out = self.base_dense(inputs)
        if self.rank and self.rank > 0 and self.lora_A is not None:
            x = inputs
            if self.dropout is not None:
                x = self.dropout(x, training=training)
            lora_out = tf.matmul(tf.matmul(x, self.lora_A), self.lora_B) * self.scaling
            out = base_out + lora_out
        else:
            out = base_out

        if self.activation is not None:
            out = self.activation(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "rank": self.rank,
                "alpha": self.alpha,
                "dropout": self.dropout_rate,
                "activation": tf.keras.activations.serialize(self.activation),
            }
        )
        return config

# ======================
# HELPERS: DICOM + PREPROCESS
# ======================
def clip_and_normalize_hu(vol, clip=CLIP_HU):
    v = np.clip(vol, clip[0], clip[1]).astype(np.float32)
    v = (v - clip[0]) / (clip[1] - clip[0] + 1e-9)
    return v

def resize_volume_to_target_phys(vol, target_shape=TARGET_SHAPE):
    """
    Simplified: just resize (Z,H,W) to target (H,W,D) via zoom.
    """
    from scipy.ndimage import zoom as ndzoom

    if vol is None:
        th, tw, td = target_shape
        return np.zeros((th, tw, td), dtype=np.float32)

    z, h, w = vol.shape
    th, tw, td = target_shape
    zoom_factors = (
        td / (z if z > 0 else 1),
        th / (h if h > 0 else 1),
        tw / (w if w > 0 else 1),
    )
    vr = ndzoom(vol, zoom_factors, order=1)  # (td, th, tw)
    vr = np.transpose(vr, (1, 2, 0))         # (H,W,D)
    return vr

def load_dicom_series_to_volume(folder: Path):
    folder = Path(folder)
    files = sorted(folder.glob("*.dcm"))
    if not files:
        files = sorted(folder.rglob("*.dcm"))
    if not files:
        return None

    candidate_slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False, force=True)
            if hasattr(ds, "pixel_array") and ds.pixel_array is not None and ds.pixel_array.size > 0:
                if ds.pixel_array.ndim == 2:
                    candidate_slices.append(ds)
        except Exception:
            continue
    if not candidate_slices:
        return None

    slices_by_shape = defaultdict(list)
    for s in candidate_slices:
        slices_by_shape[s.pixel_array.shape].append(s)
    if not slices_by_shape:
        return None

    most_common_shape = max(slices_by_shape, key=lambda shape: len(slices_by_shape[shape]))
    good_slices = slices_by_shape[most_common_shape]
    if len(good_slices) < 3:
        return None

    def zpos(s):
        try:
            return float(s.ImagePositionPatient[2])
        except Exception:
            return None

    good_slices.sort(key=lambda s: (zpos(s) is None, zpos(s), getattr(s, "InstanceNumber", 1)))

    vol = np.stack([s.pixel_array for s in good_slices]).astype(np.int16)
    for i, s in enumerate(good_slices):
        slope = float(getattr(s, "RescaleSlope", 1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        vol[i] = (vol[i].astype(np.float32) * slope + intercept).astype(np.int16)

    return vol

def choose_best_ct_series_folder(image_root: Path, patient_id: str):
    """
    Look anywhere under image_root for folders whose path contains the patient_id
    (e.g. 'KiTS-00004') and have >= 3 DICOM slices. Pick the one with most slices.
    Works for C4KC-KiTS layout: C4KC-KiTS/KiTS-00004/.../phase-folder/*.dcm
    """
    image_root = Path(image_root)
    pid = str(patient_id)

    candidates = []

    for p in image_root.rglob("*"):
        if not p.is_dir():
            continue

        # require patient id to appear somewhere in the path
        if pid not in p.as_posix():
            continue

        dcm_files = list(p.glob("*.dcm"))
        if len(dcm_files) >= 3:
            candidates.append((len(dcm_files), p))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, best_folder = candidates[0]
    return best_folder

# ======================
# LOAD MODEL + SCALERS
# ======================
@st.cache_resource
def load_inference_model():
    custom_objects = {"LoRADense": LoRADense}
    model = load_model(FULL_MODEL_PATH, custom_objects=custom_objects)
    return model

@st.cache_resource
def load_scalers():
    clin_scaler_path   = MODEL_DIR / "clin_scaler.pkl"
    clin_encoder_path  = MODEL_DIR / "clin_encoder.pkl"
    rad_scaler_path    = MODEL_DIR / "rad_scaler.pkl"

    clin_scaler  = joblib.load(clin_scaler_path)  if clin_scaler_path.exists()  else None
    clin_encoder = joblib.load(clin_encoder_path) if clin_encoder_path.exists() else None
    rad_scaler   = joblib.load(rad_scaler_path)   if rad_scaler_path.exists()   else None
    return clin_scaler, clin_encoder, rad_scaler

# ======================
# STREAMLIT APP
# ======================
st.title("C4KC Multimodal Kidney Tumor – Batch Predictor")

st.markdown("""
Upload:
1. **Clinical CSV** (C4KC-KiTS_Clinical-Data_Version.csv-style)
2. **Metadata CSV** (optional, e.g., metadata.csv)
3. **CT ZIP** with folders per patient (C4KC-KiTS/...)

The app will:
- Merge clinical + metadata
- For each patient with a CT series
- Run the multimodal model
- Output malignancy probability (and other heads if present)
""")

clinical_file = st.file_uploader("Clinical CSV", type=["csv"])
meta_file     = st.file_uploader("Metadata CSV (optional)", type=["csv"])
zip_file      = st.file_uploader("CT Image ZIP (DICOM folders)", type=["zip"])

run_btn = st.button("Run Batch Prediction")

if run_btn:
    if clinical_file is None or zip_file is None:
        st.error("Please upload at least Clinical CSV and CT ZIP.")
    else:
        with st.spinner("Loading model and scalers..."):
            model = load_inference_model()
            clin_scaler, clin_encoder, rad_scaler = load_scalers()

        # -------------------------
        # 1) Load CSVs
        # -------------------------
        clin_df = pd.read_csv(clin_file := clinical_file)
        if meta_file is not None:
            meta_df = pd.read_csv(meta_file)
        else:
            meta_df = None

        # Detect ID column (same heuristic as training)
        id_candidates = [c for c in clin_df.columns
                         if "patient" in c.lower() or "subject" in c.lower() or c.lower() == "id"]
        clinical_id_col = id_candidates[0] if id_candidates else clin_df.columns[0]

        if meta_df is not None:
            meta_id_candidates = [c for c in meta_df.columns
                                  if "patient" in c.lower() or "subject" in c.lower() or c.lower() == "id"]
            meta_id_col = meta_id_candidates[0] if meta_id_candidates else meta_df.columns[0]
            merged_df = pd.merge(
                clin_df, meta_df,
                left_on=clinical_id_col,
                right_on=meta_id_col,
                how="left"
            )
            join_key = clinical_id_col
        else:
            merged_df = clin_df.copy()
            join_key = clinical_id_col

        st.write("Merged dataframe shape:", merged_df.shape)

        # -------------------------
        # 2) Unzip CT data
        # -------------------------
        temp_root = PROJECT_ROOT / "uploaded_ct"
        if temp_root.exists():
            import shutil
            shutil.rmtree(temp_root)
        temp_root.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_file) as zf:
            zf.extractall(temp_root)

        st.write("Extracted CT data to:", temp_root)

        # Optional: show some folders for quick sanity check
        st.write("Example folders under uploaded_ct:")
        for p in list(temp_root.iterdir())[:10]:
            st.write(" -", p)

        # -------------------------
        # 3) Define feature columns
        # -------------------------
        st.markdown("Using **predefined feature column lists** from training run.")

        NUMERIC_COLS = [
            "age_at_nephrectomy",
            "body_mass_index",
            "radiographic_size",
            "pathologic_size",
            "pack_years",
        ]
        CAT_COLS = [
            # fill if you used any categorical columns
        ]

        missing_num = [c for c in NUMERIC_COLS if c not in merged_df.columns]
        if missing_num:
            st.warning(f"Some numeric columns not found in uploaded CSVs: {missing_num}")

        num_cols_used = [c for c in NUMERIC_COLS if c in merged_df.columns]
        cat_cols_used = [c for c in CAT_COLS if c in merged_df.columns]

        # -------------------------
        # 4) Build inputs per-patient
        # -------------------------
        patient_ids = merged_df[join_key].astype(str).unique().tolist()

        X_img_list = []
        X_clin_list = []
        X_rad_list = []  # currently zeros / placeholder
        pid_list = []
        preview_slices = []  # for visualization

        for pid in tqdm(patient_ids, desc="Patients", total=len(patient_ids)):
            row = merged_df.loc[merged_df[join_key].astype(str) == pid]
            if row.empty:
                continue
            row = row.iloc[0]

            # CT
            ct_folder = choose_best_ct_series_folder(temp_root, pid)
            if ct_folder is None:
                continue
            vol = load_dicom_series_to_volume(ct_folder)
            if vol is None:
                continue

            vol_proc = clip_and_normalize_hu(vol)
            vol_resz = resize_volume_to_target_phys(vol_proc)  # (H,W,D)
            if vol_resz.shape != TARGET_SHAPE:
                H, W, D = TARGET_SHAPE
                h, w, d = vol_resz.shape
                tmp = np.zeros(TARGET_SHAPE, dtype=vol_resz.dtype)
                tmp[:min(H, h), :min(W, w), :min(D, d)] = vol_resz[:min(H, h), :min(W, w), :min(D, d)]
                vol_resz = tmp

            img_tensor = np.expand_dims(vol_resz, axis=-1)  # (H,W,D,1)

            # 🔍 store mid-slice for visualization
            H, W, D = TARGET_SHAPE
            mid_idx = D // 2
            mid_slice = vol_resz[:, :, mid_idx]
            mid_slice_disp = np.clip(mid_slice, 0.0, 1.0)
            preview_slices.append(mid_slice_disp)

            # Clinical numeric
            num_vec = row[num_cols_used].astype(float).fillna(0).values if num_cols_used else np.array([0.0])

            # Clinical categorical (not really used yet since CAT_COLS is empty)
            cat_vals = [str(row[c]) if pd.notna(row[c]) else "NaN" for c in cat_cols_used]

            if clin_encoder is not None and cat_cols_used:
                cat_arr = np.array(cat_vals, dtype=object).reshape(1, -1)
                cat_encoded = clin_encoder.transform(cat_arr)
            else:
                cat_encoded = np.zeros((1, 0), dtype=np.float32)

            # Base raw vector = [num, cat_encoded]
            clin_vec_raw = np.hstack([num_vec.reshape(1, -1), cat_encoded]).astype(np.float32)  # (1, C_raw)

            # 🔧 Align feature count with what the scaler/model expect
            if clin_scaler is not None:
                n_expected = getattr(clin_scaler, "n_features_in_", clin_vec_raw.shape[1])

                if clin_vec_raw.shape[1] < n_expected:
                    # pad missing features with zeros
                    pad = np.zeros((1, n_expected - clin_vec_raw.shape[1]), dtype=clin_vec_raw.dtype)
                    clin_vec_raw = np.hstack([clin_vec_raw, pad])
                elif clin_vec_raw.shape[1] > n_expected:
                    # truncate any extra columns (shouldn’t really happen)
                    clin_vec_raw = clin_vec_raw[:, :n_expected]

                clin_vec = clin_scaler.transform(clin_vec_raw)   # → shape (1, n_expected)
            else:
                clin_vec = clin_vec_raw

            X_img_list.append(img_tensor)
            X_clin_list.append(clin_vec[0])
            pid_list.append(pid)

        if not X_img_list:
            st.error("No valid patients with CT + clinical data found. Check ZIP structure and IDs.")
        else:
            X_img_arr  = np.stack(X_img_list).astype(np.float32)   # (N,H,W,D,1)
            X_clin_arr = np.stack(X_clin_list).astype(np.float32)  # (N,C)

            # Radiomics placeholder (if model expects it)
            model_inputs = model.inputs
            use_rad = len(model_inputs) == 3
            if use_rad:
                rad_dim = int(model_inputs[2].shape[-1])
                X_rad_arr = np.zeros((len(pid_list), rad_dim), dtype=np.float32)
                inputs = [X_img_arr, X_clin_arr, X_rad_arr]
                st.info(f"Model has radiomics branch; feeding zeros of dim {rad_dim} (no radiomics computed in UI).")
            else:
                inputs = [X_img_arr, X_clin_arr]

            # -------------------------
            # 5) Run prediction
            # -------------------------
            st.write(f"Running model on {len(pid_list)} patients...")
            preds = model.predict(inputs, batch_size=1, verbose=1)

            # Handle multi-output
            if isinstance(preds, dict):
                p_main = preds["malignancy"].ravel()
                p_sub  = preds.get("subtype", None)
                p_egfr = preds.get("egfr_delta", None)
                p_read = preds.get("readmit", None)
                p_surv = preds.get("survival_risk", None)
            elif isinstance(preds, (list, tuple)):
                p_main = preds[0].ravel()
                p_sub = p_egfr = p_read = p_surv = None
            else:
                p_main = preds.ravel()
                p_sub = p_egfr = p_read = p_surv = None

            result_rows = []
            for i, pid in enumerate(pid_list):
                row_out = {
                    "patient_id": pid,
                    "malignancy_prob": float(p_main[i]),
                    "malignancy_label": int(p_main[i] >= 0.5),
                }
                result_rows.append(row_out)

            result_df = pd.DataFrame(result_rows)

            # -------------------------
            # 6) Show table + bar chart
            # -------------------------
            st.success("Prediction complete.")
            st.dataframe(result_df)

            result_df["malignancy_percent"] = 100.0 * result_df["malignancy_prob"]
            st.subheader("Malignancy probability per patient (%)")
            st.bar_chart(result_df.set_index("patient_id")["malignancy_percent"])

            # -------------------------
            # 7) Interactive CT + clinical viewer
            # -------------------------
            st.subheader("Inspect a single patient")

            selected_pid = st.selectbox(
                "Select patient to visualize",
                result_df["patient_id"].tolist()
            )

            if selected_pid in pid_list:
                idx = pid_list.index(selected_pid)

                st.write(f"**Patient:** {selected_pid}")
                prob = result_df.loc[
                    result_df["patient_id"] == selected_pid, "malignancy_prob"
                ].iloc[0]
                st.write(f"**Malignancy probability:** {prob:.2%}")

                st.image(
                    preview_slices[idx],
                    caption=f"Axial mid-slice (normalized) – {selected_pid}",
                    use_column_width=True,
                    clamp=True,
                )

                st.markdown("**Clinical summary**")
                clin_row = merged_df.loc[merged_df[join_key].astype(str) == selected_pid]

                if not clin_row.empty:
                    cols_to_show = [
                        "age_at_nephrectomy",
                        "gender",
                        "body_mass_index",
                        "radiographic_size",
                        "pathologic_size",
                        "smoking_history",
                        "pack_years",
                        "tumor_histologic_subtype",
                    ]
                    cols_to_show = [c for c in cols_to_show if c in clin_row.columns]
                    st.table(clin_row[cols_to_show])
                else:
                    st.info("No clinical row found for this patient in merged_df.")
            else:
                st.warning("Selected patient not found in pid_list – check ID alignment.")

            # -------------------------
            # 8) Download predictions
            # -------------------------
            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions as CSV",
                data=csv_bytes,
                file_name="c4kc_multimodal_predictions.csv",
                mime="text/csv"
            )
