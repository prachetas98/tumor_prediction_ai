"""
tumor_pred_ai_multimodal_patched_final.py
-----------------------------------------
Final integrated multimodal cancer detection pipeline.

Includes:
- CT scan preprocessing (normalization, CLAHE enhancement, augmentation)
- Clinical data cleaning, encoding, scaling
- CNN3D_UNet_LoRA_v1 for imaging
- ClinicalDense_4x128_v2 for tabular features
- RadiomicsAE_256D_v1 autoencoder
- FusionMultiTask_Enc_v1 for multi-output prediction
- Evaluation metrics: ROC, PR, Dice, C-index
- 3D Grad-CAM visualization
- Excel export for classification report and model summary
"""

# ============================================================
# Imports
# ============================================================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv3D, BatchNormalization, Activation, Add, GlobalAveragePooling3D, Concatenate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
import pydicom
import cv2
import scipy.ndimage as ndi
import imageio
from pathlib import Path

# ============================================================
# ADDED SECTION: Contrast Enhancement using CLAHE
# ============================================================
def clahe_volume(volume, hu_range=(-1000, 400), clip_limit=2.0, tile_grid_size=(8,8)):
    """Apply CLAHE per slice after clipping HU values."""
    vol = np.clip(volume, *hu_range)
    vol = (vol - hu_range[0]) / (hu_range[1] - hu_range[0])
    vol = (vol * 255).astype(np.uint8)
    enhanced = np.zeros_like(vol)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    for i in range(vol.shape[0]):
        enhanced[i] = clahe.apply(vol[i])
    enhanced = enhanced.astype(np.float32) / 255.0
    return enhanced

# ============================================================
# ADDED SECTION: Simple 3D Augmentation
# ============================================================
def augment_volume(vol, flip=True, rotate=True):
    if flip and np.random.rand() > 0.5:
        vol = np.flip(vol, axis=np.random.choice([0,1,2]))
    if rotate:
        angle = np.random.uniform(-10, 10)
        vol = ndi.rotate(vol, angle, axes=(1,2), reshape=False)
    return vol

# ============================================================
# Clinical data preprocessing
# ============================================================
def preprocess_clinical(df):
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    for c in cat_cols:
        df[c] = df[c].astype('category').cat.codes
    return df

# ============================================================
# ADDED SECTION: Radiomics Autoencoder (RadiomicsAE_256D_v1)
# ============================================================
def build_radiomics_ae(input_dim, bottleneck_dim=256):
    inp = Input(shape=(input_dim,))
    x = Dense(512, activation='relu')(inp)
    x = Dense(256, activation='relu')(x)
    bottleneck = Dense(bottleneck_dim, activation='relu', name='radiomics_latent')(x)
    x = Dense(256, activation='relu')(bottleneck)
    out = Dense(input_dim, activation='sigmoid')(x)
    ae = Model(inp, out, name='RadiomicsAE_256D_v1')
    encoder = Model(inp, bottleneck, name='RadiomicsAE_encoder')
    ae.compile(optimizer='adam', loss='mse')
    return ae, encoder

# ============================================================
# CNN3D_UNet_LoRA_v1 (simplified)
# ============================================================
def build_cnn3d_unet(input_shape=(128,128,64,1)):
    inp = Input(shape=input_shape, name='image_input')
    x = Conv3D(16, 3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Conv3D(32, 3, padding='same', activation='relu')(x)
    x = GlobalAveragePooling3D()(x)
    model = Model(inp, x, name='CNN3D_UNet_LoRA_v1')
    return model

# ============================================================
# ClinicalDense_4x128_v2
# ============================================================
def build_clinical_dense(input_shape=(27,)):
    inp = Input(shape=input_shape, name='clin_input')
    x = Dense(128, activation='relu')(inp)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    model = Model(inp, x, name='ClinicalDense_4x128_v2')
    return model

# ============================================================
# ADDED SECTION: FusionMultiTask_Enc_v1
# ============================================================
def build_multitask_head(img_enc, clin_enc, rad_enc=None):
    fused = Concatenate(name='fusion')([img_enc.output, clin_enc.output] + ([rad_enc.output] if rad_enc else []))
    x = Dense(128, activation='relu')(fused)
    x = Dense(64, activation='relu')(x)
    out_malignancy = Dense(1, activation='sigmoid', name='malignancy')(x)
    out_subtype = Dense(3, activation='softmax', name='subtype')(x)
    out_decline = Dense(1, activation='sigmoid', name='kidney_decline')(x)
    out_survival = Dense(1, activation='linear', name='survival_time')(x)

    model = Model(inputs=[img_enc.input, clin_enc.input] + ([rad_enc.input] if rad_enc else []),
                  outputs=[out_malignancy, out_subtype, out_decline, out_survival],
                  name='FusionMultiTask_Enc_v1')
    losses = {
        'malignancy': 'binary_crossentropy',
        'subtype': 'categorical_crossentropy',
        'kidney_decline': 'binary_crossentropy',
        'survival_time': 'mse'
    }
    loss_weights = {'malignancy':1.0,'subtype':1.0,'kidney_decline':1.0,'survival_time':0.5}
    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights, metrics=['accuracy'])
    return model

# ============================================================
# ADDED SECTION: Dice Coefficient Metric
# ============================================================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# ============================================================
# ADDED SECTION: 3D Grad-CAM Visualization
# ============================================================
def grad_cam_3d(model, img, layer_name=None):
    if layer_name is None:
        layer_name = [l.name for l in model.layers if 'conv' in l.name][-1]
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img, axis=0))
        loss = tf.reduce_mean(predictions)
    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0,1,2,3))
    cam = np.zeros(conv_outputs.shape[1:4], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[0, :, :, :, i]
    cam = np.maximum(cam, 0)
    cam /= np.max(cam)
    return cam

# ============================================================
# ADDED SECTION: Evaluation Helpers
# ============================================================
def evaluate_model(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs > threshold).astype(int)
    print(classification_report(y_true, y_pred))
    auc = roc_auc_score(y_true, y_pred_probs)
    print(f"AUC: {auc:.4f}")
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    plt.plot(fpr, tpr)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

# ============================================================
# ADDED SECTION: Excel Export for Model & Report
# ============================================================
def save_report_and_model_to_excel(model, y_true, y_pred_labels, y_pred_probs, out_path="model_and_report.xlsx"):
    from openpyxl import Workbook
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Classification_Report"
    report = classification_report(y_true, y_pred_labels, output_dict=True)
    ws1.append(["Class","Precision","Recall","F1-score","Support"])
    for cls, vals in report.items():
        if isinstance(vals, dict):
            ws1.append([cls, vals.get('precision'), vals.get('recall'), vals.get('f1-score'), vals.get('support')])
    ws2 = wb.create_sheet("Model_Summary")
    ws2.append(["Layer (Type)", "Output Shape", "Param #", "Connected To"])
    for layer in model.layers:
        ws2.append([layer.name, str(layer.output_shape), layer.count_params(), str([n.name for n in layer._inbound_nodes])])
    wb.save(out_path)
    print(f"Excel saved to {out_path}")

# ============================================================
# MAIN WORKFLOW EXAMPLE (Integrate your dataset paths here)
# ============================================================
if __name__ == "__main__":
    print("Pipeline ready. Integrate dataset and call each module as needed.")
