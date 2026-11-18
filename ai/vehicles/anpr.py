"""License plate localization and Optical Character Recognition (ANPR).

Supports:
1. Fast-ALPR (High-precision YOLO ONNX detector + MobileViT-v2 dedicated plate OCR)
2. EasyOCR (Morphological Sobel/Top-Hat plate candidate localization + CRAFT/CRNN OCR)
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .index import normalize_plate
from .schema import PlateRecord

logger = logging.getLogger("surveillance.anpr")

# Standard format regexes
INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
GENERIC_PLATE_REGEX = re.compile(r"^[A-Z0-9]{4,12}$")


def disambiguate_plate(clean: str) -> str:
    """Disambiguate common OCR character confusions (e.g. 0/O, 1/I, 8/B) based on license plate rules."""
    if len(clean) not in (9, 10):
        return clean
    if INDIAN_PLATE_REGEX.match(clean):
        return clean

    char_to_digit = {"O": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "1", "D": "0"}
    digit_to_char = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}

    chars = list(clean)
    # 1. State prefix (first 2 chars) must be letters
    for i in (0, 1):
        if chars[i].isdigit() and chars[i] in digit_to_char:
            chars[i] = digit_to_char[chars[i]]

    # 2. Registration suffix (last 4 chars) must be digits
    for i in range(len(chars) - 4, len(chars)):
        if chars[i].isalpha() and chars[i] in char_to_digit:
            chars[i] = char_to_digit[chars[i]]

    candidate = "".join(chars)
    if INDIAN_PLATE_REGEX.match(candidate):
        return candidate

    # 3. Middle district code (index 2 to 3 or 4) should be digits
    for i in range(2, 4):
        if chars[i].isalpha() and chars[i] in char_to_digit:
            chars[i] = char_to_digit[chars[i]]

    candidate = "".join(chars)
    if INDIAN_PLATE_REGEX.match(candidate):
        return candidate

    return clean


def clean_plate_text(raw_text: str) -> Tuple[str, str]:
    """Clean raw OCR text, disambiguate characters, and infer likely format."""
    clean = normalize_plate(raw_text)
    clean = disambiguate_plate(clean)
    if INDIAN_PLATE_REGEX.match(clean):
        return clean, "IND"
    if GENERIC_PLATE_REGEX.match(clean):
        return clean, "GENERIC"
    return clean, "UNKNOWN"


def _extract_box_coords(box_obj) -> Tuple[float, float, float, float]:
    """Safely extract (x1, y1, x2, y2) from various box representation formats."""
    if hasattr(box_obj, "x1") and hasattr(box_obj, "y1") and hasattr(box_obj, "x2") and hasattr(box_obj, "y2"):
        return float(box_obj.x1), float(box_obj.y1), float(box_obj.x2), float(box_obj.y2)
    if isinstance(box_obj, (list, tuple)) and len(box_obj) >= 4:
        return float(box_obj[0]), float(box_obj[1]), float(box_obj[2]), float(box_obj[3])
    if hasattr(box_obj, "tolist"):
        arr = box_obj.tolist()
        return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
    return (0.0, 0.0, 0.0, 0.0)


class LicensePlateReader:
    """Automatic Number Plate Recognition (ANPR) engine.

    Supports:
    - 'cct_s' (Default New Model): Fast-ALPR YOLOv9 + CCT-S v2 Global neural OCR (high accuracy, fixes character confusions)
    - 'cct_xs' (Legacy Model): Fast-ALPR YOLOv9 + CCT-XS v2 neural OCR (original model preserved intact)
    - 'easyocr': PyTorch CUDA CRAFT detector + deep text recognition
    - 'mobile_vit': Fast-ALPR YOLOv9 + MobileViT-v2 transformer OCR
    """

    def __init__(self, engine: str = "auto", languages: list[str] = ["en"], gpu: bool = True):
        """Initialize ANPR engine.

        Parameters
        ----------
        engine : 'auto', 'cct_s', 'cct_xs', 'legacy', 'easyocr', or 'mobile_vit'
        languages : languages for fallback EasyOCR
        gpu : whether to enable GPU acceleration
        """
        import os
        env_engine = os.environ.get("ANPR_ENGINE")
        if env_engine:
            engine = env_engine

        self.requested_engine = (engine or "cct_s").lower().strip()
        self._languages = languages
        self._gpu = gpu
        self._fast_alpr = None
        self._easyocr_reader = None
        self._active_ocr_model = ""
        self.active_engine = "none"

        self.set_engine(self.requested_engine)

    def _init_fast_alpr(self, ocr_model: str = "cct-s-v2-global-model") -> bool:
        """Attempt to load the Fast-ALPR neural pipeline with specified OCR model."""
        try:
            from fast_alpr import ALPR
            self._fast_alpr = ALPR(
                detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model=ocr_model,
            )
            self._active_ocr_model = ocr_model
            logger.info("Initialized Fast-ALPR neural engine with OCR model: %s", ocr_model)
            return True
        except ImportError:
            logger.debug("Fast-ALPR not installed. Falling back to EasyOCR.")
        except Exception as e:
            logger.warning("Fast-ALPR init with %s notice: %s. Falling back to EasyOCR.", ocr_model, e)
        return False

    def set_engine(self, engine: str) -> str:
        """Switch active OCR engine at runtime without destroying existing models.

        Options: 'cct_s' (new default), 'cct_xs' (legacy), 'easyocr', 'mobile_vit'.
        """
        eng = (engine or "cct_s").lower().strip()
        self.requested_engine = eng

        if eng in ("cct_xs", "legacy", "fast_alpr_legacy"):
            if self._init_fast_alpr(ocr_model="cct-xs-v2-global-model"):
                self.active_engine = "cct_xs (legacy Fast-ALPR)"
            else:
                self.active_engine = "easyocr"
        elif eng in ("mobile_vit", "mvit"):
            if self._init_fast_alpr(ocr_model="global-plates-mobile-vit-v2-model"):
                self.active_engine = "mobile_vit"
            else:
                self.active_engine = "easyocr"
        elif eng == "easyocr":
            if self.easyocr_reader is not None:
                self.active_engine = "easyocr"
            else:
                self._init_fast_alpr(ocr_model="cct-s-v2-global-model")
                self.active_engine = "cct_s (Fast-ALPR v2)"
        else:
            # Default: 'cct_s' or 'auto' or 'fast_alpr'
            if self._init_fast_alpr(ocr_model="cct-s-v2-global-model"):
                self.active_engine = "cct_s (Fast-ALPR v2)"
            elif self._init_fast_alpr(ocr_model="cct-xs-v2-global-model"):
                self.active_engine = "cct_xs (legacy Fast-ALPR)"
            else:
                self.active_engine = "easyocr"

        return self.active_engine

    @property
    def fast_alpr(self):
        """Getter for fast_alpr instance."""
        if self._fast_alpr is None and self.requested_engine != "easyocr":
            self.set_engine(self.requested_engine)
        return self._fast_alpr

    @property
    def easyocr_reader(self):
        """Lazy load EasyOCR reader fallback."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(self._languages, gpu=self._gpu, verbose=False)
            except Exception as e:
                logger.warning("EasyOCR init notice: %s", e)
                self._easyocr_reader = False
        return self._easyocr_reader if self._easyocr_reader is not False else None

    def locate_plates(self, vehicle_img: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Find candidate license plate bounding boxes in a vehicle crop using morphological features."""
        if vehicle_img is None or vehicle_img.size == 0:
            return []

        h, w = vehicle_img.shape[:2]
        if h < 20 or w < 30:
            return []

        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuring_element)

        sobel_x = cv2.Sobel(top_hat, cv2.CV_32F, 1, 0, ksize=3)
        sobel_x = np.absolute(sobel_x)
        min_val, max_val = np.min(sobel_x), np.max(sobel_x)
        if max_val > min_val:
            sobel_x = (255 * (sobel_x - min_val) / (max_val - min_val)).astype(np.uint8)
        else:
            sobel_x = np.zeros_like(gray)

        blurred = cv2.GaussianBlur(sobel_x, (5, 5), 0)
        close_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, close_elem)
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        veh_area = h * w

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch == 0:
                continue
            aspect = cw / float(ch)
            area = cw * ch
            if 1.8 <= aspect <= 5.8 and (0.008 * veh_area <= area <= 0.35 * veh_area):
                candidates.append((float(x), float(y), float(x + cw), float(y + ch)))

        if candidates:
            candidates.sort(key=lambda b: -b[1])
            return candidates[:3]

        # Fallback: bumper region
        fallback_y1 = float(h * 0.65)
        fallback_h = float(h * 0.3)
        fallback_x1 = float(w * 0.15)
        fallback_w = float(w * 0.7)
        return [(fallback_x1, fallback_y1, fallback_x1 + fallback_w, fallback_y1 + fallback_h)]

    def read_text_easyocr(self, plate_crop: np.ndarray) -> Optional[Tuple[str, float]]:
        """Run EasyOCR on cropped license plate."""
        reader = self.easyocr_reader
        if plate_crop is None or plate_crop.size == 0 or reader is None:
            return None

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
        resized = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)

        results = reader.readtext(enhanced)
        if not results:
            return None

        all_text = " ".join([r[1] for r in results])
        confs = [r[2] for r in results]
        avg_conf = float(sum(confs) / max(1, len(confs)))

        clean, fmt = clean_plate_text(all_text)
        if len(clean) >= 4:
            return clean, avg_conf
        return None

    def process_vehicle(self, frame: np.ndarray, vehicle_bbox: tuple) -> Optional[PlateRecord]:
        """Detect and recognize license plate for a single vehicle bounding box."""
        x1, y1, x2, y2 = (int(round(v)) for v in vehicle_bbox)
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        veh_crop = frame[y1:y2, x1:x2]

        # -------------------------------------------------------------
        # 1. EasyOCR Requested Engine: Dedicated PyTorch CRAFT / CRNN
        # -------------------------------------------------------------
        if self.requested_engine == "easyocr":
            # If Fast-ALPR detector is available, use it for precise plate box localization
            if self._fast_alpr is not None:
                try:
                    rgb_crop = cv2.cvtColor(veh_crop, cv2.COLOR_BGR2RGB)
                    det_res = self._fast_alpr.predict(rgb_crop)
                    for r in det_res:
                        box_obj = getattr(r.detection, "bounding_box", None) or getattr(r, "box", None)
                        bx1, by1, bx2, by2 = _extract_box_coords(box_obj)
                        p_crop = veh_crop[max(0, int(by1)):int(by2), max(0, int(bx1)):int(bx2)]
                        if p_crop.size > 0:
                            easy_res = self.read_text_easyocr(p_crop)
                            if easy_res:
                                clean, e_conf = easy_res
                                return PlateRecord(
                                    text=clean,
                                    confidence=e_conf,
                                    bbox=(float(x1 + bx1), float(y1 + by1), float(x1 + bx2), float(y1 + by2)),
                                    country_format="IND" if INDIAN_PLATE_REGEX.match(clean) else "GENERIC",
                                )
                except Exception as e:
                    logger.debug("EasyOCR detector crop error: %s", e)

            # Fallback to morphological plate localization
            candidates = self.locate_plates(veh_crop)
            for px1, py1, px2, py2 in candidates:
                c_x1, c_y1 = max(0, int(round(px1))), max(0, int(round(py1)))
                c_x2, c_y2 = min(veh_crop.shape[1], int(round(px2))), min(veh_crop.shape[0], int(round(py2)))
                if c_x2 <= c_x1 or c_y2 <= c_y1:
                    continue
                p_crop = veh_crop[c_y1:c_y2, c_x1:c_x2]
                easy_res = self.read_text_easyocr(p_crop)
                if easy_res:
                    clean, e_conf = easy_res
                    return PlateRecord(
                        text=clean,
                        confidence=e_conf,
                        bbox=(float(x1 + c_x1), float(y1 + c_y1), float(x1 + c_x2), float(y1 + c_y2)),
                        country_format="IND" if INDIAN_PLATE_REGEX.match(clean) else "GENERIC",
                    )
            return None

        # -------------------------------------------------------------
        # 2. Neural ALPR Engine (cct_s, cct_xs, or mobile_vit)
        # -------------------------------------------------------------
        if self.fast_alpr is not None:
            try:
                # Fast-ALPR expects RGB numpy arrays
                rgb_crop = cv2.cvtColor(veh_crop, cv2.COLOR_BGR2RGB)
                alpr_results = self.fast_alpr.predict(rgb_crop)
                if not alpr_results and frame is not None:
                    # Fallback to full frame plate detection
                    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    full_results = self.fast_alpr.predict(rgb_full)
                    if full_results:
                        for r in full_results:
                            if r.ocr and r.ocr.text:
                                clean, fmt = clean_plate_text(r.ocr.text)
                                if len(clean) >= 4:
                                    box_obj = getattr(r.detection, "bounding_box", None) or getattr(r, "box", None)
                                    bx1, by1, bx2, by2 = _extract_box_coords(box_obj)
                                    c_val = r.ocr.confidence
                                    final_conf = float(c_val) if isinstance(c_val, (int, float)) else float(sum(c_val) / max(1, len(c_val)))
                                    return PlateRecord(
                                        text=clean,
                                        confidence=final_conf,
                                        bbox=(bx1, by1, bx2, by2),
                                        country_format=fmt,
                                    )

                if alpr_results:
                    # Select detection with highest combined confidence
                    best_res = None
                    best_score = -1.0
                    for r in alpr_results:
                        if r.ocr and r.ocr.text:
                            c_ocr = r.ocr.confidence
                            ocr_val = float(c_ocr) if isinstance(c_ocr, (int, float)) else float(sum(c_ocr) / max(1, len(c_ocr)))
                            det_val = float(getattr(r.detection, "confidence", 1.0))
                            combined = ocr_val * det_val
                            if combined > best_score:
                                best_score = combined
                                best_res = r

                    if best_res and best_res.ocr and best_res.ocr.text:
                        raw_text = best_res.ocr.text
                        clean, fmt = clean_plate_text(raw_text)
                        if len(clean) >= 4:
                            # Extract bounding box
                            box_obj = getattr(best_res.detection, "bounding_box", None) or getattr(best_res, "box", None)
                            bx1, by1, bx2, by2 = _extract_box_coords(box_obj)
                            global_bbox = (
                                float(x1 + bx1),
                                float(y1 + by1),
                                float(x1 + bx2),
                                float(y1 + by2),
                            )
                            c_val = best_res.ocr.confidence
                            final_conf = float(c_val) if isinstance(c_val, (int, float)) else float(sum(c_val) / max(1, len(c_val)))
                            return PlateRecord(
                                text=clean,
                                confidence=final_conf,
                                bbox=global_bbox,
                                country_format=fmt,
                            )
            except Exception as e:
                logger.debug("Fast-ALPR vehicle pass note: %s", e)

        # -------------------------------------------------------------
        # 2. Secondary Fallback: Morphological Filtering + EasyOCR
        # -------------------------------------------------------------
        candidates = self.locate_plates(veh_crop)
        for px1, py1, px2, py2 in candidates:
            c_x1, c_y1 = max(0, int(round(px1))), max(0, int(round(py1)))
            c_x2, c_y2 = min(veh_crop.shape[1], int(round(px2))), min(veh_crop.shape[0], int(round(py2)))
            if c_x2 <= c_x1 or c_y2 <= c_y1:
                continue

            plate_crop = veh_crop[c_y1:c_y2, c_x1:c_x2]
            res = self.read_text_easyocr(plate_crop)
            if res is not None:
                text, conf = res
                clean, fmt = clean_plate_text(text)
                global_bbox = (
                    float(x1 + c_x1),
                    float(y1 + c_y1),
                    float(x1 + c_x2),
                    float(y1 + c_y2),
                )
                return PlateRecord(
                    text=clean,
                    confidence=conf,
                    bbox=global_bbox,
                    country_format=fmt,
                )

        return None

    def read_frame(self, frame: np.ndarray) -> List[PlateRecord]:
        """Detect and read all license plates directly across the full frame using Fast-ALPR."""
        if frame is None or frame.size == 0:
            return []

        records = []
        if self.fast_alpr is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.fast_alpr.predict(rgb_frame)
                for r in results:
                    if r.ocr and r.ocr.text:
                        clean, fmt = clean_plate_text(r.ocr.text)
                        if len(clean) >= 4:
                            box_obj = getattr(r.detection, "bounding_box", None) or getattr(r, "box", None)
                            bx1, by1, bx2, by2 = _extract_box_coords(box_obj)
                            c_val = r.ocr.confidence
                            final_conf = float(c_val) if isinstance(c_val, (int, float)) else float(sum(c_val) / max(1, len(c_val)))
                            records.append(
                                PlateRecord(
                                    text=clean,
                                    confidence=final_conf,
                                    bbox=(bx1, by1, bx2, by2),
                                    country_format=fmt,
                                )
                            )
            except Exception as e:
                logger.debug("Fast-ALPR full-frame pass note: %s", e)
        return records

