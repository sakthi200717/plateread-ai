import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageStat

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
NOISY_DIR = DATASET / "noisy_plates"
LABELS_FILE = DATASET / "labels.csv"

st.set_page_config(page_title="PlateRead AI | Vehicle Investigation", page_icon="🚘", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --accent:#d8232a; --accent-dark:#b91c22; --ink:#242a35; --muted:#707887; --line:#e8e8eb; --surface:#ffffff; }
.stApp { background:#f7f7f8; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(255,255,255,.92); }
.block-container { padding-top:1.2rem; padding-bottom:2rem; max-width:1500px; }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid #e9e9ec; }
[data-testid="stSidebar"] * { color:#343a46 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#6e7480 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:7px 10px; border-radius:8px; }
.brand { font-size:1.12rem; font-weight:850; letter-spacing:-.025em; color:var(--accent) !important; padding:8px 4px 2px; }
.brand-sub { color:#858a94 !important; font-size:.66rem; letter-spacing:.13em; margin:0 4px 26px; }
.eyebrow { color:var(--accent); font-size:.72rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
.page-title { font-size:1.7rem; font-weight:800; color:#242a35; margin:0; letter-spacing:-.035em; }
.page-sub { color:#777e89; font-size:.9rem; margin-top:4px; }
.panel { background:#fff; border:1px solid var(--line); border-radius:14px; padding:18px 20px; box-shadow:0 2px 10px rgba(30,35,45,.025); }
.panel-title { color:#242a35; font-size:1rem; font-weight:780; margin-bottom:4px; }
.panel-sub { color:#7a808a; font-size:.78rem; margin-bottom:14px; }
.status-online { color:#18835d; background:#e8f7ef; border-radius:20px; padding:6px 11px; font-size:.72rem; font-weight:750; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:12px 14px; border-radius:12px; box-shadow:0 2px 10px rgba(30,35,45,.025); }
.stButton > button, .stDownloadButton > button { border-radius:8px; font-weight:700; border:1px solid #dedfe3; transition:all .15s ease; }
.stButton > button[kind="primary"] { background:var(--accent); border-color:var(--accent); color:#fff; }
.stButton > button[kind="primary"]:hover { background:var(--accent-dark); border-color:var(--accent-dark); }
.stDownloadButton > button:hover, .stButton > button:hover { border-color:var(--accent); color:var(--accent); }
[data-testid="stFileUploader"] { background:#fff; border:1px dashed #d5d7dc; border-radius:12px; padding:8px; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div { border-radius:8px; }
hr { border-color:#ececef; }
</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown('<div class="brand">PlateRead AI</div><div class="brand-sub">SMART NUMBER PLATE ANALYSIS</div>', unsafe_allow_html=True)
    nav = st.radio("NAVIGATION", ["Home", "Datasets", "Used Dataset", "Detected Plates", "Profile"], label_visibility="collapsed")
    st.markdown("<div style='height:170px'></div><div style='font-size:.75rem;color:#91a4ba'>OCR ENGINE STATUS</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#6ee7ad;font-size:.8rem'>● Ready to analyze</div>", unsafe_allow_html=True)
    st.caption("PlateRead AI · Demo build")
st.markdown('<div style="display:flex;justify-content:space-between;align-items:center"><div><div class="eyebrow">PLATEREAD AI / SMART DETECTION</div><div class="page-title">Number Plate Search & Analysis</div><div class="page-sub">Upload or select a noisy plate image, run OCR, and review detection results.</div></div><div class="status-online">● SYSTEM READY</div></div>', unsafe_allow_html=True)
st.markdown(" ")

if shutil.which("tesseract") is None:
    candidate = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if candidate.exists():
        pytesseract.pytesseract.tesseract_cmd = str(candidate)

if not NOISY_DIR.exists() or not LABELS_FILE.exists():
    st.error("Dataset is missing. Keep the dataset folder next to app.py.")
    st.stop()

labels = pd.read_csv(LABELS_FILE)
required = {"image", "reference_number", "severity"}
if not required.issubset(labels.columns):
    st.error(f"labels.csv needs columns: {sorted(required)}")
    st.stop()

st.markdown("#### Choose an input image")
uploaded_file = st.file_uploader("Upload a number-plate image (JPG, JPEG, or PNG)",
                                 type=["jpg", "jpeg", "png"], key="plate_image_upload",
                                 help="Uploaded images are analyzed directly. No reference label is assumed.")
if uploaded_file is not None:
    st.caption(f"Using uploaded image: {uploaded_file.name}")

top1, top2, top3 = st.columns([2.4, 1.2, 1.2])
with top1:
    selected_row_index = st.selectbox("OR SELECT A DATASET IMAGE", options=list(range(len(labels))),
        format_func=lambda i: f"{Path(str(labels.iloc[i]['image'])).name} · {labels.iloc[i]['severity']} blur")
with top2:
    mode = st.selectbox("OCR ENGINE", ["Automatic (recommended)", "EasyOCR only", "Tesseract only"])
with top3:
    st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
    st.markdown(f"<span class='status-online'>● {len(labels)} DATASET IMAGES</span>", unsafe_allow_html=True)
row = labels.iloc[selected_row_index]
if uploaded_file is not None:
    selected_path = Path(uploaded_file.name)
    original = Image.open(uploaded_file).convert("RGB")
    reference_number = ""
    severity_value = "uploaded"
    uploaded_size_kb = len(uploaded_file.getvalue()) / 1024
else:
    selected_path = DATASET / str(row["image"])
    original = Image.open(selected_path).convert("RGB")
    reference_number = str(row.get("reference_number", "") or "").strip()
    severity_value = str(row.get("severity", "unknown"))
    uploaded_size_kb = None

def attrs_for(img, path, uploaded_size_kb=None):
    gray = ImageOps.grayscale(img)
    stats = ImageStat.Stat(gray)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    if uploaded_size_kb is not None:
        size_kb = uploaded_size_kb
    elif path.exists():
        size_kb = path.stat().st_size / 1024
    else:
        size_kb = 0
    return {
        "Width (px)": img.width,
        "Height (px)": img.height,
        "Brightness": round(stats.mean[0], 2),
        "Contrast": round(stats.stddev[0], 2),
        "Edge variance": round(float(ImageStat.Stat(edges).var[0]), 2),
        "File size (KB)": round(size_kb, 2),
    }

def build_variants(img):
    gray = ImageOps.grayscale(img)
    # Include full image and a generous crop around the character line.
    w,h=gray.size
    crops = [
        ("full", gray),
        ("plate-body", gray.crop((int(w*.12), int(h*.08), int(w*.995), int(h*.94)))),
        ("text-area", gray.crop((int(w*.13), int(h*.14), int(w*.995), int(h*.88)))),
    ]
    variants=[]
    for crop_name,crop in crops:
        for scale in (2,3,4):
            up=crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.LANCZOS)
            contrast=ImageOps.autocontrast(up)
            variants.append((f"{crop_name} x{scale} gray",up))
            variants.append((f"{crop_name} x{scale} contrast",contrast))
            variants.append((f"{crop_name} x{scale} sharp",contrast.filter(
                ImageFilter.UnsharpMask(radius=1.3,percent=170,threshold=2))))
            if CV2_AVAILABLE:
                arr=np.array(contrast)
                den=cv2.fastNlMeansDenoising(arr,None,7,7,21)
                variants.append((f"{crop_name} x{scale} denoise",Image.fromarray(den)))
                for block,c in ((21,5),(31,9)):
                    th=cv2.adaptiveThreshold(den,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY,block,c)
                    variants.append((f"{crop_name} x{scale} adaptive{block}",Image.fromarray(th)))
                    variants.append((f"{crop_name} x{scale} adaptive{block} inv",Image.fromarray(255-th)))
    return variants

def clean_candidate(text):
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())

def normalize_plate(text):
    compact=clean_candidate(text)
    # Match expected format within a larger OCR string.
    m=re.search(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}",compact)
    if m:
        return m.group(0)
    # Position-aware correction as a fallback; no labels used in prediction.
    to_letters=str.maketrans({"0":"O","1":"I","5":"S","8":"B"})
    to_digits=str.maketrans({"O":"0","D":"0","Q":"0","I":"1","L":"1","Z":"2","S":"5","B":"8","G":"6"})
    for rto_len in (2,1):
        for series_len in (3,2,1):
            for num_len in (4,3,2,1):
                if len(compact)!=2+rto_len+series_len+num_len:
                    continue
                state=compact[:2].translate(to_letters)
                rto=compact[2:2+rto_len].translate(to_digits)
                p=2+rto_len
                series=compact[p:p+series_len].translate(to_letters)
                number=compact[p+series_len:].translate(to_digits)
                if state.isalpha() and rto.isdigit() and series.isalpha() and number.isdigit():
                    return state+rto+series+number
    return ""

@st.cache_resource
def get_reader():
    if not EASYOCR_AVAILABLE:
        return None
    return easyocr.Reader(["en"],gpu=False,verbose=False)

def easyocr_candidates(img):
    reader=get_reader()
    if reader is None:
        return [], ["EasyOCR not installed."]
    found=[]
    diagnostics=[]
    for name,variant in build_variants(img)[:4]: # fast mode: limit repeated OCR passes
        try:
            results=reader.readtext(np.array(variant),detail=1,paragraph=False,
                                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            for _bbox,txt,conf in results:
                raw=clean_candidate(txt)
                plate=normalize_plate(raw)
                if raw:
                    diagnostics.append(f"EasyOCR {name}: {raw} ({conf:.2f})")
                if plate:
                    found.append({"plate":plate,"confidence":float(conf)*100,
                                  "method":f"EasyOCR {name}","engine":"EasyOCR"})
        except Exception:
            continue
    return found,diagnostics

def tesseract_candidates(img):
    found=[]
    diagnostics=[]
    configs=["--oem 1 --psm 7","--oem 1 --psm 8"]
    for name,variant in build_variants(img)[:3]:
        for config in configs:
            try:
                cfg=config+" -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                raw=pytesseract.image_to_string(variant,config=cfg)
                if raw.strip():
                    diagnostics.append(f"Tesseract {name} {config}: {raw.strip()}")
                plate=normalize_plate(raw)
                data=pytesseract.image_to_data(variant,config=cfg,output_type=pytesseract.Output.DICT)
                tokens=[]; scores=[]
                for token,conf in zip(data.get("text",[]),data.get("conf",[])):
                    token=clean_candidate(token)
                    if token: tokens.append(token)
                    try:
                        val=float(conf)
                        if val>=0: scores.append(val)
                    except (ValueError,TypeError): pass
                if not plate:
                    plate=normalize_plate("".join(tokens))
                if plate:
                    found.append({"plate":plate,
                                  "confidence":sum(scores)/len(scores) if scores else 0,
                                  "method":f"Tesseract {name} {config}","engine":"Tesseract"})
            except pytesseract.pytesseract.TesseractNotFoundError:
                diagnostics.append("Tesseract executable not found.")
                break
            except Exception:
                continue
    return found,diagnostics

def predict(img, recognition_mode):
    candidates=[]; diagnostics=[]
    if recognition_mode in ("Automatic (recommended)","EasyOCR only"):
        easy,diag=easyocr_candidates(img); candidates.extend(easy); diagnostics.extend(diag[:8])
    if recognition_mode in ("Automatic (recommended)","Tesseract only"):
        tess,diag=tesseract_candidates(img); candidates.extend(tess); diagnostics.extend(diag[:8])
    if not candidates:
        return "",0.0,"No plate candidate",diagnostics
    grouped={}
    for c in candidates:
        grouped.setdefault(c["plate"],[]).append(c)
    ranked=[]
    for plate,items in grouped.items():
        engines=len(set(x["engine"] for x in items))
        avg=sum(x["confidence"] for x in items)/len(items)
        ranked.append({"plate":plate,"engines":engines,"votes":len(items),"confidence":avg,
                       "best":max(items,key=lambda x:x["confidence"])})
    ranked.sort(key=lambda x:(x["engines"],x["votes"],x["confidence"]),reverse=True)
    best=ranked[0]
    return best["plate"],round(best["confidence"],1),f'{best["best"]["engine"]} · {best["best"]["method"]}',diagnostics

# Shared dataset navigation pages. All tables are derived from labels.csv and actual session activity.
page = nav
if page == "Home":
    st.markdown("<div class='panel-title'>CURRENT IMAGE ANALYSIS</div><div class='panel-sub'>Original image remains unchanged; enhancement is applied only for OCR.</div>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Original noisy image</div><div class="panel-sub">Selected dataset image · preview</div>', unsafe_allow_html=True)
        st.image(original, caption=selected_path.name, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("Enhanced preview for inspection"):
            enhanced = ImageOps.autocontrast(ImageOps.grayscale(original)).filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
            st.image(enhanced, caption="Preview only — original is not modified", use_container_width=True)
    with right:
        st.markdown('<div class="panel-title">Image quality attributes</div><div class="panel-sub">Input image diagnostics</div>', unsafe_allow_html=True)
        attrs = attrs_for(original, selected_path, uploaded_size_kb)
        c1, c2 = st.columns(2)
        for i, (key, value) in enumerate(attrs.items()):
            (c1 if i % 2 == 0 else c2).metric(key, str(value))
        st.markdown("<div class='panel-title' style='margin-top:18px'>OCR detection</div>", unsafe_allow_html=True)
        if not EASYOCR_AVAILABLE:
            st.info("EasyOCR is not installed; Automatic mode will use Tesseract if available.")
        if not CV2_AVAILABLE:
            st.caption("OpenCV is unavailable; denoising/adaptive threshold variants are disabled.")
        if st.button("🔎 Analyze plate", type="primary", use_container_width=True):
            with st.spinner("Running optimized OCR checks. First run may take longer while the model initializes..."):
                try:
                    prediction, confidence, method, diagnostics = predict(original, mode)
                    st.session_state["last_result"] = {"image": selected_path.name, "prediction": prediction,
                        "confidence": confidence, "method": method, "diagnostics": diagnostics, "mode": mode,
                        "reference": reference_number, "severity": severity_value, "attrs": attrs}
                    history = st.session_state.get("used_dataset", [])
                    history = [x for x in history if x.get("image") != selected_path.name]
                    history.append({"image": selected_path.name, "prediction": prediction or "Not detected",
                        "confidence": confidence, "method": method, "reference": reference_number,
                        "severity": severity_value, "exact_match": bool(reference_number and prediction and clean_candidate(prediction) == clean_candidate(reference_number))})
                    st.session_state["used_dataset"] = history
                except Exception as exc:
                    st.error(f"OCR failed: {exc}")
    result = st.session_state.get("last_result")
    if result and result["image"] == selected_path.name:
        st.markdown("<div class='panel-title'>DETECTION RESULT</div>", unsafe_allow_html=True)
        if result["prediction"]: st.success("OCR candidate found — verify it against the original image.")
        else: st.warning("No registration candidate detected. Review the image or try another OCR engine.")
        a, b, c = st.columns(3)
        a.metric("Predicted plate", result["prediction"] or "Not detected")
        b.metric("OCR score", f'{result["confidence"]}%')
        c.metric("Blur severity", result["severity"].title())
        st.caption(f'Method: {result["method"]}. The score is not a guarantee of correctness.')
        if str(result.get("reference", "")).strip():
            match = bool(result["prediction"] and clean_candidate(result["prediction"]) == clean_candidate(result["reference"]))
            st.metric("Reference comparison (evaluation only)", "Exact match" if match else "Mismatch / not detected")
        else:
            match = False
            st.caption("Uploaded image: no reference label supplied, so exact-match evaluation is not available.")
        with st.expander("OCR diagnostics"):
            if result["diagnostics"]:
                for line in result["diagnostics"][:16]: st.text(line)
            else: st.write("No diagnostics returned.")
        report = {"Image": result["image"], "Severity": result["severity"], "Predicted plate": result["prediction"] or "Not detected",
                  "Reference label (evaluation only)": result["reference"] or "Not provided", "Exact match": match if result["reference"] else "Not evaluated",
                  "Confidence score": result["confidence"], "Method": result["method"], **result["attrs"]}
        df = pd.DataFrame([report])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download selected result CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="plate_ocr_result.csv", mime="text/csv")

elif page == "Datasets":
    st.markdown("<div class='page-title'>Complete Dataset</div><div class='page-sub'>Browse every image listed in dataset/labels.csv.</div>", unsafe_allow_html=True)
    m1,m2,m3 = st.columns(3)
    m1.metric("Total labelled images", len(labels))
    m2.metric("Unique reference plates", labels["reference_number"].nunique())
    m3.metric("Severity levels", labels["severity"].nunique())
    severity_filter = st.multiselect("Filter by blur severity", sorted(labels["severity"].astype(str).unique()), default=sorted(labels["severity"].astype(str).unique()))
    filtered = labels[labels["severity"].astype(str).isin(severity_filter)]
    st.caption(f"Showing {len(filtered)} of {len(labels)} dataset images")
    st.dataframe(filtered.rename(columns={"image":"Image path", "reference_number":"Reference label (evaluation only)", "severity":"Severity", "synthetic":"Synthetic sample", "clean_image":"Clean reference image"}), use_container_width=True, hide_index=True)
    st.download_button("Download full dataset index (CSV)", labels.to_csv(index=False).encode("utf-8"), file_name="platerread_full_dataset_index.csv", mime="text/csv")
    st.markdown("#### Image gallery")
    gallery_count = st.slider("Images to preview", min_value=4, max_value=min(24, max(4, len(filtered))), value=min(12, max(4, len(filtered))), step=4)
    gallery = filtered.head(gallery_count)
    cols = st.columns(4)
    for idx, (_, item) in enumerate(gallery.iterrows()):
        img_path = DATASET / str(item["image"])
        with cols[idx % 4]:
            if img_path.exists(): st.image(Image.open(img_path), caption=f'{img_path.name} · {item["severity"]}', use_container_width=True)
            st.caption("Reference: " + str(item["reference_number"]))

elif page == "Used Dataset":
    st.markdown("<div class='page-title'>Used Dataset</div><div class='page-sub'>Images you have actually analyzed during this app session.</div>", unsafe_allow_html=True)
    used = st.session_state.get("used_dataset", [])
    if not used:
        st.info("No images analyzed in this session yet. Go to Home, select an image and click Analyze plate.")
    else:
        used_df = pd.DataFrame(used)
        a,b,c = st.columns(3)
        a.metric("Images analyzed", len(used_df))
        c.metric("Exact matches", int(used_df["exact_match"].sum()))
        b.metric("Not detected", int((used_df["prediction"] == "Not detected").sum()))
        st.dataframe(used_df.rename(columns={"image":"Image", "prediction":"Prediction", "confidence":"OCR score", "method":"Method", "reference":"Reference (evaluation only)", "severity":"Severity", "exact_match":"Exact match"}), use_container_width=True, hide_index=True)
        st.download_button("Download used dataset history", used_df.to_csv(index=False).encode("utf-8"), file_name="platerread_used_dataset.csv", mime="text/csv")
        if st.button("Clear used history"):
            st.session_state["used_dataset"] = []
            st.rerun()

elif page == "Detected Plates":
    st.markdown("<div class='page-title'>Detected Plates</div><div class='page-sub'>Search predictions from this session.</div>", unsafe_allow_html=True)
    query = st.text_input("Search plate number", placeholder="Enter a predicted plate string")
    used = pd.DataFrame(st.session_state.get("used_dataset", []))
    if used.empty:
        st.info("Analyze images from Home to populate detected plates.")
    else:
        shown = used[used["prediction"].astype(str).str.contains(query.strip(), case=False, na=False)] if query.strip() else used
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.download_button("Download detected plates CSV", shown.to_csv(index=False).encode("utf-8"), file_name="detected_plates.csv", mime="text/csv")

elif page == "Profile":
    st.markdown("<div class='page-title'>Dataset Profile</div><div class='page-sub'>Overview of the complete dataset included with this project.</div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total dataset images", len(labels))
    c2.metric("Unique plate labels", labels["reference_number"].nunique())
    c3.metric("Analyzed this session", len(st.session_state.get("used_dataset", [])))
    c4.metric("Image severity groups", labels["severity"].nunique())
    st.markdown("#### Full dataset inventory")
    st.dataframe(labels.rename(columns={"image":"Image path", "reference_number":"Reference label (evaluation only)", "severity":"Severity", "synthetic":"Synthetic sample", "clean_image":"Clean reference image"}), use_container_width=True, hide_index=True)
    st.download_button("Download all dataset records", labels.to_csv(index=False).encode("utf-8"), file_name="platerread_profile_full_dataset.csv", mime="text/csv")
    st.caption("Predictions are generated from images. Reference labels are shown for dataset inventory/evaluation only.")

st.divider()
st.markdown("<div style='text-align:center;color:#8792a2;font-size:.75rem'>PLATEREAD AI · Smart OCR workspace · Always verify predictions against the original image.</div>", unsafe_allow_html=True)
