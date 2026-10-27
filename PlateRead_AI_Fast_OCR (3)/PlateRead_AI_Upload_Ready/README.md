# PlateRead AI — Robust Noisy Plate OCR

## Included
- 20 clean synthetic demo plates.
- 60 paired noisy variants: light, medium, and heavy blur/noise.
- `dataset/labels.csv` maps each noisy image to its known demo plate for evaluation.
- Streamlit dashboard with original noisy image display, image metrics, EasyOCR + Tesseract, preprocessing variants, raw OCR diagnostics, and CSV export.
- Clean images are provided only as references; predictions are produced from noisy images only.

## Run on Windows
Open this folder (the one containing `app.py` and `requirements.txt`) in VS Code, then run:
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
If you already installed the requirements, run only the last command. Do not run the command from the outer ZIP folder; run it from this project folder.
First EasyOCR use downloads pretrained model weights and can take time. Tesseract is an additional recognizer; on Windows the app checks `C:\Program Files\Tesseract-OCR\tesseract.exe`.

## Important
Images are synthetic, not real vehicle records. The code does not use reference labels to generate predictions. It uses them only after inference to evaluate exact matches. OCR confidence is not a guarantee. Severe blur may erase character details, making correct recovery impossible.


## Added four noisy sample images

Four separate noisy plate samples are included as `dataset/noisy_plates/user_sample_1.png` through `user_sample_4.png`. The reference strings for generated samples are provisional and must be visually verified before using exact-match evaluation. The app predicts from image pixels first; the reference labels are used only for comparison after prediction. OCR may still misread blurred characters, so check each output against the original image. These are generated examples, not verified real vehicle records.


## Upload your own image
On the Home page, use **Upload a number-plate image** to choose a JPG/JPEG/PNG file. Uploaded images are analyzed directly. The app does not assume a ground-truth label for uploads, so exact-match evaluation is shown only for dataset items that have a reference label. A heavily blurred sample is included at `dataset/noisy_plates/uploaded_style_heavy_blur.png`.

## Dashboard navigation
- **Datasets** shows the complete dataset inventory and image gallery.
- **Used Dataset** records images analyzed during the current Streamlit session.
- **Detected Plates** searches predictions from the current session.
- **Profile** shows totals and the full dataset inventory.
- Reference labels are for evaluation only and are never passed into the OCR prediction function.
