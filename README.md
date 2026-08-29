# License_Plate_OCR

Flask server for the graduation project — detects Egyptian license plates and extracts Arabic letters + numbers.

- **Input:** `POST /upload` with `image` file
- **Output:** `{"message": ["text1", "text2"]}` from PaddleOCR (`lang='ar'`)
- **Stack:** YOLO (`best.pt`), OpenCV, PaddleOCR, EDSR x4 super-resolution (`EDSR_x4.pb`)

## Quick start

```bash
pip install -r requirements.txt  # flask ultralytics opencv-python paddleocr numpy requests
python app.py                    # -> http://127.0.0.1:5000
python client.py ./A_hamdy.png   # test upload
```

## How it works

1. YOLO detects plate, Hough `Canny`+`HoughLinesP` estimates skew, rotate whole image.
2. Re-run YOLO on deskewed image for tighter box, crop with small negative margin, preprocess to gray 10:3 strip.
3. Single EDSR x4 upsample, then `PaddleOCR(lang='ar')` (loaded once globally).

## Branches

- `main` — original
- `cleanup/lean-2026` — cleaned demos/artifacts, optimized EDSR 4x (was 16x double), OCR global load

## Notes

- Generated `rec_neoplate_before/after.jpg` are gitignored debug dumps.
- `tester.py` is batch eval (no Flask) — kept as legacy until `app.py` verified.
