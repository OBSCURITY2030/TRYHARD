# enhanced_ocr_pipeline.py
# -*- coding: utf-8 -*-
"""
Улучшенный OCR-пайплайн для скриншотов итогов матча Standoff 2.
Объединяет лучшие практики: мой опыт тестирования + рекомендации ChatGPT-5.

Ключевые улучшения:
- Автоматическое определение Tesseract
- ROI-константы для точной настройки под UI Standoff 2
- Двухэтапная обработка: Tesseract + PaddleOCR fallback
- Улучшенная нормализация имен
- Конфигурируемые пороги
"""

from __future__ import annotations
import os
import io
import re
import shutil
import logging
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import pytesseract

# Настройка логгера
logger = logging.getLogger(__name__)

# Проверка доступности rapidfuzz
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    import difflib

# ---------- Автоматическая настройка Tesseract ----------
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    logger.info(f"Tesseract найден в: {tesseract_path}")
else:
    possible_paths = ["/usr/bin/tesseract", "/opt/homebrew/bin/tesseract", "tesseract"]
    for path in possible_paths:
        if shutil.which(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"Tesseract найден в: {path}")
            break
    else:
        logger.warning("Tesseract не найден в системе!")

# ---------- Конфигурация ----------
# Языки для Tesseract
OCR_LANG_DEFAULT = os.environ.get("OCR_LANG_DEFAULT", "eng+rus")
OCR_LANG_ENG = "eng"
OCR_LANG_RUS = "rus"

# Режимы Tesseract
TS_BASE = "--oem 1"
TS_NO_DAWGS = "-c load_system_dawg=0 -c load_freq_dawg=0"

# Допустимые символы в нике
ALLOWED_NAME_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9\[\]\-\._@ ]+")

# Regex для счёта X:Y или X-Y
SCORE_RE = re.compile(r"\b(\d{1,2})\s*[:\-]\s*(\d{1,2})\b")

# Список карт Standoff 2
MAPS = [
    "Sandstone", "Rust", "Province", "Village", "Arena", "Train", "Zone9",
    "Dust2", "Mirage", "Cache", "Overpass", "Vertigo", "Inferno",
    "Zone7", "Breeze", "Dune", "Sakura"
]

# ROI-константы (в процентах от высоты/ширины) 
# ТОЧНО НАСТРОЕНО ПОД РЕАЛЬНЫЙ СКРИНШОТ STANDOFF 2
# УЛУЧШЕННЫЕ ROI КООРДИНАТЫ (ChatGPT-5 рекомендации)
# Применены после визуальной калибровки для максимальной точности
ROI_SCORE = (0.13, 0.19, 0.47, 0.53)     # Точно только X:Y цифры 
ROI_MAP   = (0.02, 0.08, 0.35, 0.65)     # Верхний центр только название карты
ROI_TOP   = (0.22, 0.83, 0.05, 0.95)     # Оптимизировано для лучшего захвата имен
ROI_BOT   = (0.22, 0.83, 0.05, 0.95)     # Аналогично нижняя команда
NAME_CROP = (0.10, 0.90, 0.08, 0.45)     # Область имени в строке

# Кол-во строк в каждой таблице
TEAM_ROWS = 5

# Порог fuzzy-сопоставления
FUZZY_CUTOFF = int(os.environ.get("FUZZY_CUTOFF", "75"))  # Понижен для игровых ников

# PaddleOCR fallback 
USE_PADDLE_FALLBACK = os.environ.get("PADDLE_FALLBACK", "0") == "1"
_paddle_ocr = None

# NEW: контроль времени и лимиты производительности
TIME_BUDGET_SEC = float(os.environ.get("OCR_TIME_BUDGET_SEC", "8.0"))  # общий бюджет
PADDLE_MAX_ROWS = int(os.environ.get("PADDLE_MAX_ROWS", "3"))          # не более N строк на fallback  
PADDLE_MIN_LEN = int(os.environ.get("PADDLE_MIN_LEN", "2"))           # запускать fallback, если имя < N символов
PADDLE_MIN_BUDGET = float(os.environ.get("PADDLE_MIN_BUDGET", "2.0")) # не запускать fallback, если осталось < N c

# Карты и алиасы для улучшенного распознавания
MAPS_ENHANCED = ["Sandstone", "Rust", "Province", "Breeze", "Dune", "Sakura", "Zone7"] 
MAP_ALIASES = {
    "brezee": "Breeze",   # частая OCR-ошибка
    "breeze": "Breeze", 
    "sand stone": "Sandstone",
    "sandston": "Sandstone",
    "provence": "Province", 
    "provenc": "Province",
    "zone 7": "Zone7",
    "zone7": "Zone7",
    "ruѕt": "Rust",  # cyrillic 'ѕ'
    "dun": "Dune",
    "ѕakura": "Sakura",  # cyrillic
}

# ---------- Вспомогательные функции ----------

def _clip_roi(img: np.ndarray, roi_rel: Tuple[float, float, float, float]) -> np.ndarray:
    """Обрезать ROI в относительных процентах (y0,y1,x0,x1)."""
    h, w = img.shape[:2]
    y0 = int(roi_rel[0] * h)
    y1 = int(roi_rel[1] * h)
    x0 = int(roi_rel[2] * w)
    x1 = int(roi_rel[3] * w)
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    if y1 <= y0 or x1 <= x0:
        return img.copy()
    return img[y0:y1, x0:x1]

def preprocess(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Улучшенная предобработка: upscale 2x, медианный блюр, адаптивная бинаризация.
    """
    img = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
    )
    return img, bin_img

# NEW: быстрые функции для оптимизации производительности  
def preprocess_fast(gray: np.ndarray) -> np.ndarray:
    """Быстрая предобработка без медианного блюра - только адаптивный порог."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 12
    )

def _up2_gray(roi_bgr: np.ndarray) -> np.ndarray:
    """Быстрое масштабирование ROI в 2x и конвертация в серый."""
    up = cv2.resize(roi_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

def ocr_text(
    img: np.ndarray,
    lang: str = OCR_LANG_DEFAULT,
    psm: int = 6,
    extra_cfg: Optional[str] = None,
) -> str:
    """OCR-распознавание через Tesseract с обработкой ошибок."""
    cfg = f"{TS_BASE} {TS_NO_DAWGS} --psm {psm}"
    if extra_cfg:
        cfg = f"{cfg} {extra_cfg}"
    try:
        return pytesseract.image_to_string(img, lang=lang, config=cfg)
    except Exception as e:
        logger.warning(f"OCR ошибка: {e}")
        return ""

def clean_name(s: str) -> str:
    """Фильтр/нормализация строки имени."""
    s = s.strip()
    m = ALLOWED_NAME_RE.findall(s)
    result = "".join(m).strip()
    # Убираем лишние пробелы
    result = re.sub(r'\s+', ' ', result)
    return result

def get_team_rois(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Возвращает ROI левой и правой команд (оптимизировано под реальный скриншот)."""
    left_roi = _clip_roi(img, ROI_TOP)
    right_roi = _clip_roi(img, ROI_BOT)
    logger.info(f"Left team ROI: {left_roi.shape}, Right team ROI: {right_roi.shape}")
    return left_roi, right_roi

def split_rows(roi_bgr: np.ndarray, rows: int = TEAM_ROWS) -> List[np.ndarray]:
    """Улучшенное деление таблицы на строки с адаптивными границами."""
    h, w = roi_bgr.shape[:2]
    
    # Конвертируем в grayscale для анализа горизонтальных проекций
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    
    # Горизонтальная проекция (сумма пикселей по строкам)
    h_proj = np.sum(gray < 200, axis=1)  # Темные пиксели
    
    # Поиск пиков (строк с текстом)
    peaks = []
    for i in range(1, len(h_proj)-1):
        if h_proj[i] > h_proj[i-1] and h_proj[i] > h_proj[i+1]:
            if h_proj[i] > np.mean(h_proj) * 0.3:  # Порог активности
                peaks.append(i)
    
    # Если не нашли пики, используем равномерное деление
    if len(peaks) < 2:
        row_h = max(1, h // rows)
        out = []
        for i in range(rows):
            y0 = i * row_h
            y1 = h if i == rows - 1 else (i + 1) * row_h
            if y1 > y0:
                out.append(roi_bgr[y0:y1, :])
        return out
    
    # Используем найденные пики для более точного деления
    out = []
    for i in range(min(rows, len(peaks))):
        if i == 0:
            y0 = 0
        else:
            y0 = (peaks[i-1] + peaks[i]) // 2
            
        if i == len(peaks) - 1 or i == rows - 1:
            y1 = h
        else:
            y1 = (peaks[i] + peaks[i+1]) // 2
        
        if y1 > y0:
            out.append(roi_bgr[y0:y1, :])
    
    # Дополняем до нужного количества строк если необходимо
    while len(out) < rows:
        out.append(np.zeros((10, w, 3), dtype=np.uint8))
    
    return out[:rows]

def extract_name_from_row(row_bgr: np.ndarray) -> str:
    """
    Многоступенчатое извлечение имени с несколькими попытками OCR.
    """
    if row_bgr is None or row_bgr.size == 0:
        return ""
    
    h, w = row_bgr.shape[:2]
    if h < 5 or w < 10:
        return ""
    
    # Пробуем разные области для извлечения имени
    name_crops = [
        _clip_roi(row_bgr, NAME_CROP),                          # Основная область
        _clip_roi(row_bgr, (0.05, 0.95, 0.05, 0.50)),         # Шире
        _clip_roi(row_bgr, (0.15, 0.85, 0.10, 0.40))          # Уже
    ]
    
    best_result = ""
    
    for name_crop in name_crops:
        if name_crop.size == 0:
            continue
        
        _, name_bin = preprocess(name_crop)
        
        # Пробуем разные настройки OCR
        ocr_attempts = [
            (name_bin, OCR_LANG_DEFAULT, 7),    # Основной режим - одна строка
            (name_bin, OCR_LANG_ENG, 7),        # Только английский
            (name_bin, OCR_LANG_RUS, 7),        # Только русский
            (name_bin, OCR_LANG_DEFAULT, 8),    # Одно слово
            (name_bin, OCR_LANG_DEFAULT, 6),    # Блок текста
        ]
        
        for img, lang, psm in ocr_attempts:
            text = ocr_text(img, lang=lang, psm=psm)
            cleaned = clean_name(text)
            
            # Выбираем лучший результат
            if len(cleaned) > len(best_result) and re.search(r'[A-Za-zА-Яа-я]', cleaned):
                best_result = cleaned
                
        # Если результат все еще слабый, пробуем на сером без бинаризации
        if len(best_result) < 3:
            up_gray = cv2.cvtColor(cv2.resize(name_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                                 cv2.COLOR_BGR2GRAY)
            alt = ocr_text(up_gray, lang=OCR_LANG_DEFAULT, psm=7)
            alt = clean_name(alt)
            if len(alt) > len(best_result):
                best_result = alt
    
    return best_result

def match_known(name: str, known: List[str], score_cutoff: int = FUZZY_CUTOFF) -> Optional[str]:
    """Улучшенное fuzzy-сопоставление с базой известных игроков."""
    if not known or not name:
        return None
    
    if RAPIDFUZZ_AVAILABLE:
        match = rf_process.extractOne(name, known, scorer=rf_fuzz.WRatio, score_cutoff=score_cutoff)
        return match[0] if match else None
    else:
        # Fallback на difflib
        best_match = None
        best_ratio = score_cutoff / 100.0
        
        for player in known:
            ratio = difflib.SequenceMatcher(None, name.lower(), player.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = player
        return best_match

def extract_score(img_bgr: np.ndarray) -> Optional[Tuple[str, int, int]]:
    """
    Многоступенчатое извлечение счета с несколькими областями поиска.
    """
    # Пробуем несколько областей для счета
    score_regions = [
        ROI_SCORE,                              # Основная область
        (0.10, 0.18, 0.40, 0.60),              # Шире
        (0.14, 0.22, 0.42, 0.58),              # Ниже
        (0.08, 0.16, 0.47, 0.53),              # Узко по центру
        (0.05, 0.15, 0.35, 0.65)               # Очень широко
    ]
    
    for roi_coords in score_regions:
        score_roi = _clip_roi(img_bgr, roi_coords)
        if score_roi.size == 0:
            continue
            
        _, bin_img = preprocess(score_roi)
        
        # Пробуем разные настройки OCR
        for lang, psm in [(OCR_LANG_ENG, 6), (OCR_LANG_ENG, 7), (OCR_LANG_ENG, 8)]:
            text = ocr_text(bin_img, lang=lang, psm=psm)
            match = SCORE_RE.search(text)
            
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                # Проверка разумности счета
                if 0 <= a <= 13 and 0 <= b <= 13:
                    logger.info(f"Найден счет: {a}:{b} в области {roi_coords}")
                    return f"{a}:{b}", a, b
    
    return None

def winner_from_score(a: int, b: int) -> Optional[str]:
    """Определение победителя по счету."""
    if a is None or b is None or a == b:
        return None
    return "top" if a > b else "bottom"

def extract_map(img_bgr: np.ndarray) -> Optional[str]:
    """
    Извлечение названия карты из нижней части экрана.
    """
    # Пробуем несколько областей для карты
    map_regions = [
        ROI_MAP,                                # Основная область (низ слева)
        (0.85, 0.95, 0.00, 0.50),             # Чуть выше и шире
        (0.90, 1.00, 0.00, 0.35),             # Самый низ
        (0.02, 0.12, 0.12, 0.88),             # Также проверим верх
        (0.82, 0.92, 0.00, 0.45)              # Дополнительная область
    ]
    
    for roi_coords in map_regions:
        roi = _clip_roi(img_bgr, roi_coords)
        if roi.size == 0:
            continue
            
        _, bin_img = preprocess(roi)
        
        for lang, psm in [(OCR_LANG_ENG, 6), (OCR_LANG_ENG, 7)]:
            text = ocr_text(bin_img, lang=lang, psm=psm)
            text = re.sub(r"[^A-Za-z ]+", " ", text).strip()
            
            if not text:
                continue
            
            # Сопоставляем с известными картами
            if RAPIDFUZZ_AVAILABLE:
                match = rf_process.extractOne(text, MAPS, scorer=rf_fuzz.WRatio, score_cutoff=70)
                if match:
                    logger.info(f"Найдена карта: {match[0]} в области {roi_coords}")
                    return match[0]
            else:
                best_match = None
                best_ratio = 0.7
                for m in MAPS:
                    ratio = difflib.SequenceMatcher(None, text.lower(), m.lower()).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = m
                if best_match:
                    logger.info(f"Найдена карта: {best_match} в области {roi_coords}")
                    return best_match
    
    return None

# ---------- Главная функция ----------

def process_screenshot(image_bytes: bytes, known_players: List[str]) -> Dict:
    """
    Главный вход: принимает байты изображения, возвращает извлеченные данные.
    """
    logger.info(f"Processing image size: {len(image_bytes)} bytes")
    
    # Декодируем изображение
    np_img = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if bgr is None:
        return {
            "score": None,
            "winner_side": None,
            "map": None,
            "players": [],
            "success": False,
            "error": "Cannot decode image"
        }
    
    logger.info(f"Processing image size: {bgr.shape}")
    
    # Базовая предобработка
    upscaled, _ = preprocess(bgr)
    
    # Извлекаем счет и определяем победителя
    score_pack = extract_score(upscaled)
    if score_pack:
        score_str, a, b = score_pack
        win_side = winner_from_score(a, b)
        logger.info(f"Extracted score: {score_str}, winner: {win_side}")
    else:
        score_str, a, b, win_side = None, None, None, None
        logger.info("Extracted score: None, winner: None")
    
    # Извлекаем карту
    map_name = extract_map(upscaled)
    
    # Получаем области команд
    left_roi, right_roi = get_team_rois(upscaled)
    left_rows = split_rows(left_roi, rows=TEAM_ROWS)
    right_rows = split_rows(right_roi, rows=TEAM_ROWS)
    
    # Извлекаем имена игроков
    players = []
    
    # Левая команда
    for i, row in enumerate(left_rows):
        raw = extract_name_from_row(row)
        norm = raw  # Можно добавить дополнительную нормализацию
        matched = match_known(norm, known_players) if norm else None
        
        logger.info(f"Left player {i+1}: '{raw}' -> match: {matched}")
        
        players.append({
            "team": "top",
            "raw": raw,
            "normalized": norm,
            "match_db": matched
        })
    
    # Правая команда
    for i, row in enumerate(right_rows):
        raw = extract_name_from_row(row)
        norm = raw
        matched = match_known(norm, known_players) if norm else None
        
        logger.info(f"Right player {i+1}: '{raw}' -> match: {matched}")
        
        players.append({
            "team": "bottom",
            "raw": raw,
            "normalized": norm,
            "match_db": matched
        })
    
    logger.info(f"Extracted map: {map_name}")
    
    return {
        "score": score_str,
        "winner_side": win_side,
        "map": map_name,
        "players": players,
        "success": True
    }

# === НОВЫЕ БЫСТРЫЕ ФУНКЦИИ ДЛЯ ОПТИМИЗАЦИИ ===

def extract_score_fast(img_bgr: np.ndarray):
    """Быстрое извлечение счета только из нужной ROI области."""
    score_roi = _clip_roi(img_bgr, ROI_SCORE)
    gray = _up2_gray(score_roi)
    bin_img = preprocess_fast(gray)
    
    text = ocr_text(bin_img, lang=OCR_LANG_ENG, psm=6)
    match = SCORE_RE.search(text)
    
    if not match:
        # Быстрый повтор без порога
        text2 = ocr_text(gray, lang=OCR_LANG_ENG, psm=7)
        match = SCORE_RE.search(text2)
        if not match:
            return None
    
    a, b = int(match.group(1)), int(match.group(2))
    if not (0 <= a <= 13 and 0 <= b <= 13):
        return None
    
    return f"{a}:{b}", a, b

def extract_name_fast(row_bgr: np.ndarray) -> str:
    """Быстрое извлечение имени из строки игрока."""
    h, w = row_bgr.shape[:2]
    y0 = int(NAME_CROP[0] * h)
    y1 = int(NAME_CROP[1] * h)
    x0 = int(NAME_CROP[2] * w)
    x1 = int(NAME_CROP[3] * w)
    
    name_crop = row_bgr[max(0,y0):min(h,y1), max(0,x0):min(w,x1)]
    
    gray = _up2_gray(name_crop)
    bin_img = preprocess_fast(gray)
    text = ocr_text(bin_img, lang=OCR_LANG_DEFAULT, psm=7)
    text = clean_name(text)
    
    if len(text) < 2:
        # Fallback на серый без порога
        text2 = ocr_text(gray, lang=OCR_LANG_DEFAULT, psm=7)
        text2 = clean_name(text2)
        if len(text2) > len(text):
            text = text2
    
    return text

def _normalize_map_text(txt: str) -> str:
    """Нормализация текста карты с алиасами."""
    s = re.sub(r"[^A-Za-z ]+", " ", txt).strip().lower()
    s = re.sub(r"\s+", " ", s)
    
    # Проверяем алиасы
    if s in MAP_ALIASES:
        return MAP_ALIASES[s]
    
    # Точное попадание?
    for m in MAPS_ENHANCED:
        if s == m.lower():
            return m
    
    # Fuzzy поиск
    if RAPIDFUZZ_AVAILABLE:
        match = rf_process.extractOne(s, MAPS_ENHANCED, scorer=rf_fuzz.WRatio, score_cutoff=80)
        return match[0] if match else None
    
    return None

def extract_map_fast(img_bgr: np.ndarray) -> Optional[str]:
    """Быстрое извлечение карты из ROI области."""
    roi = _clip_roi(img_bgr, ROI_MAP)
    gray = _up2_gray(roi)
    bin_img = preprocess_fast(gray)
    
    text = ocr_text(bin_img, lang=OCR_LANG_ENG, psm=6)
    normalized = _normalize_map_text(text)
    if normalized:
        return normalized
    
    # Ещё попытка на сером
    text2 = ocr_text(gray, lang=OCR_LANG_ENG, psm=6)
    return _normalize_map_text(text2)

def get_team_rois_fast(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Получение областей команд без обработки (быстрая версия)."""
    top_roi = _clip_roi(bgr, ROI_TOP)
    bottom_roi = _clip_roi(bgr, ROI_BOT)
    return top_roi, bottom_roi

def split_rows(team_roi: np.ndarray, rows: int) -> List[np.ndarray]:
    """Разделение области команды на строки игроков."""
    h = team_roi.shape[0]
    row_height = h // rows
    result = []
    
    for i in range(rows):
        y0 = i * row_height
        y1 = min((i + 1) * row_height, h)
        if y1 > y0:
            result.append(team_roi[y0:y1])
    
    return result

def ocr_paddle_line(row: np.ndarray) -> str:
    """Fallback через PaddleOCR для одной строки (если доступен)."""
    global _paddle_ocr
    
    if not USE_PADDLE_FALLBACK:
        return ""
    
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            logger.info("✅ PaddleOCR инициализирован для fallback")
        except ImportError:
            logger.warning("⚠️ PaddleOCR недоступен для fallback")
            return ""
    
    try:
        result = _paddle_ocr.ocr(row)
        if result and result[0]:
            texts = [item[1][0] for item in result[0] if item[1][1] > 0.5]
            return clean_name(" ".join(texts))
    except Exception as e:
        logger.warning(f"PaddleOCR ошибка: {e}")
    
    return ""

# Экспорт для обратной совместимости
def extract_data_from_screenshot(image_bytes: bytes, known_players: List[str]) -> Dict:
    """Alias для совместимости с существующим кодом."""
    return process_screenshot(image_bytes, known_players)