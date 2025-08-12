#!/usr/bin/env python3
"""
УЛЬТРА-БЫСТРАЯ OCR система - решение проблемы скорости.
Цель: 1-3 секунды + ПОЛНАЯ информация за счет минимальных OCR вызовов.

Стратегия: 3 OCR вызова = ВСЯ информация
1. Счет (1 вызов)
2. Карта (1 вызов)  
3. Все игроки (1 вызов на весь скриншот)
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
import pytesseract
import re

logger = logging.getLogger(__name__)

# ROI координаты из Enhanced OCR (проверенные)
ROI_SCORE = (0.13, 0.19, 0.47, 0.53)
ROI_MAP = (0.02, 0.08, 0.35, 0.65)
ROI_TOP = (0.22, 0.83, 0.05, 0.95)
ROI_BOT = (0.22, 0.83, 0.05, 0.95)

# Карты Standoff 2
STANDOFF_MAPS = ["Breeze", "Rust", "Province", "Zone7", "Sandstone", "Sakura", "Dune", "Village", "Arena", "Train"]

# Ультра-быстрые настройки Tesseract
FAST_CONFIG = "--oem 3 -c user_defined_dpi=150"

def _clip_roi(img: np.ndarray, roi_rel: Tuple[float, float, float, float]) -> np.ndarray:
    """Обрезка ROI области."""
    h, w = img.shape[:2]
    y0, y1, x0, x1 = roi_rel
    y0, y1 = int(y0 * h), int(y1 * h)
    x0, x1 = int(x0 * w), int(x1 * w)
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    if y1 <= y0 or x1 <= x0:
        return img.copy()
    return img[y0:y1, x0:x1]

def ultra_fast_ocr(img, lang="eng", psm=6, whitelist=""):
    """Ультра-быстрый OCR - минимальная конфигурация."""
    config = f"{FAST_CONFIG} --psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(img, lang=lang, config=config).strip()

def extract_score_ultra_fast(img_bgr):
    """1 OCR вызов для счета."""
    roi = _clip_roi(img_bgr, ROI_SCORE)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Минимальная обработка - только resize
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 1 OCR вызов с whitelist только для цифр
    text = ultra_fast_ocr(resized, lang="eng", psm=7, whitelist="0123456789:")
    
    # Поиск счета
    match = re.search(r'(\d{1,2})\s*:\s*(\d{1,2})', text)
    if match:
        left, right = match.groups()
        score = f"{left}:{right}"
        winner = "top" if int(left) > int(right) else "bottom" if int(right) > int(left) else "draw"
        return score, winner
    
    return None, "unknown"

def extract_map_ultra_fast(img_bgr):
    """1 OCR вызов для карты."""
    roi = _clip_roi(img_bgr, ROI_MAP)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Минимальная обработка
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 1 OCR вызов с whitelist для букв
    text = ultra_fast_ocr(resized, lang="eng", psm=6, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ")
    
    # Быстрый поиск карт
    text_lower = text.lower()
    for map_name in STANDOFF_MAPS:
        if map_name.lower() in text_lower:
            return map_name
    
    # Алиасы
    if "breez" in text_lower:
        return "Breeze"
    if "zone" in text_lower:
        return "Zone7"
    if "sand" in text_lower:
        return "Sandstone"
    
    return None

def extract_all_players_ultra_fast(img_bgr):
    """1 OCR вызов для ВСЕХ игроков сразу."""
    # Берем области команд
    top_roi = _clip_roi(img_bgr, ROI_TOP)
    bottom_roi = _clip_roi(img_bgr, ROI_BOT)
    
    # Объединяем в один большой ROI
    combined_height = top_roi.shape[0] + bottom_roi.shape[0] + 50  # +50 для разделителя
    combined_width = max(top_roi.shape[1], bottom_roi.shape[1])
    
    # Создаем объединенное изображение
    combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    # Размещаем команды
    combined[:top_roi.shape[0], :top_roi.shape[1]] = top_roi
    combined[top_roi.shape[0]+50:top_roi.shape[0]+50+bottom_roi.shape[0], :bottom_roi.shape[1]] = bottom_roi
    
    # Преобразуем в grayscale и увеличиваем
    gray = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # 1 OCR вызов на ВСЕ имена
    text = ultra_fast_ocr(resized, lang="eng", psm=6, 
                         whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя0123456789[]-._@# ")
    
    # Парсим результат
    lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 1]
    
    # Очистка имен
    cleaned_names = []
    for line in lines:
        # Базовая очистка
        cleaned = re.sub(r'[^\w\[\]\-\._@# ]', '', line)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Исправления OCR
        cleaned = cleaned.replace('0', 'O').replace('1', 'I').replace('5', 'S').replace('8', 'B')
        
        if len(cleaned) > 1:
            cleaned_names.append(cleaned)
    
    # Распределяем по командам (первые 5 - top, остальные - bottom)
    top_players = cleaned_names[:5] + [""] * (5 - len(cleaned_names[:5]))
    bottom_players = cleaned_names[5:10] + [""] * (5 - len(cleaned_names[5:10]))
    
    return top_players, bottom_players

def simple_fuzzy_match(name, known_players, cutoff=75):
    """Простое fuzzy matching без внешних библиотек."""
    if not name or not known_players:
        return None
    
    name_lower = name.lower()
    best_match = None
    best_score = 0
    
    for player in known_players:
        player_lower = player.lower()
        
        # Точное совпадение
        if name_lower == player_lower:
            return player
        
        # Частичное совпадение
        if name_lower in player_lower or player_lower in name_lower:
            score = min(len(name_lower), len(player_lower)) / max(len(name_lower), len(player_lower)) * 100
            if score > best_score and score >= cutoff:
                best_score = score
                best_match = player
    
    return best_match

def process_screenshot_ultra_fast(image_bytes: bytes, known_players: List[str] = None) -> Dict:
    """
    УЛЬТРА-БЫСТРАЯ обработка: 3 OCR вызова = ВСЯ информация.
    Цель: 1-3 секунды.
    """
    start_time = time.time()
    
    try:
        # Декодирование изображения
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return {"success": False, "error": "Cannot decode image"}
        
        if not known_players:
            known_players = []
        
        logger.info(f"Ultra-fast processing: {img_bgr.shape}, known: {len(known_players)}")
        
        # === 3 OCR ВЫЗОВА = ВСЯ ИНФОРМАЦИЯ ===
        
        # 1. Счет (1 OCR вызов)
        score, winner = extract_score_ultra_fast(img_bgr)
        
        # 2. Карта (1 OCR вызов)
        map_name = extract_map_ultra_fast(img_bgr)
        
        # 3. Все игроки (1 OCR вызов)
        top_names, bottom_names = extract_all_players_ultra_fast(img_bgr)
        
        # Объединяем игроков
        all_names = top_names + bottom_names
        teams = ["top"] * 5 + ["bottom"] * 5
        
        # Быстрое сопоставление
        players = []
        for i, (name, team) in enumerate(zip(all_names, teams)):
            if name:
                matched = simple_fuzzy_match(name, known_players if known_players else [], cutoff=80)
                players.append({
                    "raw": name,
                    "normalized": name,
                    "matched": matched,
                    "team": team,
                    "row": i % 5,
                    "confidence": 90 if matched else 70
                })
        
        processing_time = time.time() - start_time
        
        # Оценка качества
        has_score = bool(score and ":" in str(score))
        has_map = bool(map_name)
        has_enough_players = len(players) >= 8
        
        success_rate = 0.0
        if has_score:
            success_rate += 0.3
        if has_map:
            success_rate += 0.3  
        if has_enough_players:
            success_rate += 0.4
        
        result = {
            "success": True,
            "score": score,
            "winner_side": winner,
            "map": map_name,
            "players": players,
            "processing_time": round(processing_time, 2),
            "method": "UltraFastOCR",
            "phase": "УЛЬТРА-БЫСТРАЯ: 3 OCR вызова",
            "performance": {
                "target_achieved": processing_time <= 3.0,
                "players_found": len(players),
                "success_rate": round(success_rate, 3),
                "has_score": has_score,
                "has_map": has_map,
                "data_completeness": "ПОЛНАЯ" if success_rate >= 0.8 else "ЧАСТИЧНАЯ",
                "ocr_calls": 3  # Всего 3 вызова OCR!
            }
        }
        
        logger.info(f"✅ Ultra-fast: {processing_time:.2f}s, {len(players)} players, score: {score}, map: {map_name}")
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Ultra-fast error: {e}")
        
        return {
            "success": False,
            "error": str(e),
            "processing_time": round(processing_time, 2),
            "method": "UltraFastOCR"
        }

if __name__ == "__main__":
    print("⚡ УЛЬТРА-БЫСТРАЯ OCR система готова")
    print("Цель: 1-3 секунды через 3 OCR вызова")