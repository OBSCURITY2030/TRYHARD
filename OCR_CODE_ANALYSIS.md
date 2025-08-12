# OCR CODE ANALYSIS - ПОЛНЫЙ КОД ДЛЯ CHATGPT

## ОСНОВНОЙ TELEGRAM BOT КОД (advanced_bot_simple.py - OCR секция):

```python
# === ВЫБОР ЛУЧШЕЙ OCR СИСТЕМЫ ===
def get_best_ocr_function():
    """Выбор лучшего доступного OCR пайплайна."""
    if ULTRA_FAST_AVAILABLE:
        return process_screenshot_ultra_fast, "⚡ УЛЬТРА-БЫСТРАЯ: 3 OCR вызова (1-3s)"
    elif FIXED_BATCH_AVAILABLE:
        return process_screenshot_fixed, "ИСПРАВЛЕНА: Быстро + Полные данные (2-5s)"
    elif BATCH_PIPELINE_AVAILABLE:
        return process_screenshot_batch, "Batch Pipeline (2-5s target)"
    else:
        return process_screenshot, "Enhanced Pipeline (8-10s)"

# === ОБРАБОТКА СКРИНШОТА В БОТЕ ===
@dp.message_handler(content_types=['photo'])
async def handle_screenshot(message: types.Message):
    try:
        # Скачиваем фото
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        file_bytes = await bot.download_file(file_info.file_path)
        
        processing_msg = await message.answer("🔍 Обрабатываю скриншот...")
        
        # Получаем известных игроков
        known_players = list(users.keys())
        
        # ОСНОВНАЯ OCR ОБРАБОТКА
        ocr_function, method_name = get_best_ocr_function()
        result = ocr_function(file_bytes.read(), known_players)
        
        # Парсим результат
        if result.get('success'):
            await process_ocr_result(message, result, processing_msg)
        else:
            await processing_msg.edit_text("❌ Ошибка обработки скриншота")
            
    except Exception as e:
        logger.error(f"Screenshot processing error: {e}")
        await message.answer("❌ Произошла ошибка при обработке")
```

## ПРОБЛЕМНАЯ ULTRA FAST OCR СИСТЕМА:

```python
#!/usr/bin/env python3
"""
УЛЬТРА-БЫСТРАЯ OCR система - НЕ РАБОТАЕТ КАК ОЖИДАЛОСЬ
Проблема: extract_score занимает 6.99 секунд вместо <1 секунды
"""

def extract_score_ultra_fast(img_bgr):
    """1 OCR вызов для счета - МЕДЛЕННЫЙ!"""
    roi = _clip_roi(img_bgr, ROI_SCORE)  # (0.13, 0.19, 0.47, 0.53)
    
    # Минимальная обработка - без предобработки, напрямую
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # ОДИН вызов OCR, максимально быстро - НО ЗАНИМАЕТ 6.99с!
    text = ultra_fast_ocr(gray, lang="eng", psm=7, extra_cfg=TS_WL_SCORE)
    
    # Поиск счета
    score_match = re.search(r'(\d{1,2})\s*[:\-]\s*(\d{1,2})', text)
    
    if score_match:
        left, right = score_match.groups()
        score = f"{left}:{right}"
        winner = "top" if int(left) > int(right) else "bottom" if int(right) > int(left) else "draw"
        return score, winner
    
    return None, "unknown"

def ultra_fast_ocr(img, lang="eng", psm=6, whitelist=""):
    """Ультра-быстрый OCR - НО ВСЕ РАВНО МЕДЛЕННЫЙ"""
    config = f"{FAST_CONFIG} --psm {psm}"  # "--oem 3 -c user_defined_dpi=150"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(img, lang=lang, config=config).strip()
```

## ENHANCED OCR (РАБОТАЕТ, НО МЕДЛЕННО):

```python
# enhanced_ocr_pipeline.py - МНОЖЕСТВЕННЫЕ OCR ВЫЗОВЫ
def extract_name_from_row(row_bgr: np.ndarray) -> str:
    """
    Многоступенчатое извлечение имени с несколькими попытками OCR.
    ПРОБЛЕМА: 5+ OCR вызовов на каждое имя = 20+ вызовов на скриншот
    """
    # Пробуем разные области для извлечения имени
    name_crops = [
        _clip_roi(row_bgr, NAME_CROP),                          # Основная область
        _clip_roi(row_bgr, (0.05, 0.95, 0.05, 0.50)),         # Шире
        _clip_roi(row_bgr, (0.15, 0.85, 0.10, 0.40))          # Уже
    ]
    
    for name_crop in name_crops:
        # Пробуем разные настройки OCR - ВОТ ПРОБЛЕМА!
        ocr_attempts = [
            (name_bin, OCR_LANG_DEFAULT, 7),    # Основной режим
            (name_bin, OCR_LANG_ENG, 7),        # Только английский
            (name_bin, OCR_LANG_RUS, 7),        # Только русский
            (name_bin, OCR_LANG_DEFAULT, 8),    # Одно слово
            (name_bin, OCR_LANG_DEFAULT, 6),    # Блок текста
        ]
        
        for img, lang, psm in ocr_attempts:  # 5 попыток на каждое имя!
            text = ocr_text(img, lang=lang, psm=psm)
```

## BATCH OPTIMIZED (БЫСТРЫЙ, НО НЕ РАБОТАЕТ):

```python
# batch_optimized_pipeline.py - Теряет данные при batch обработке
def extract_team_names_batch(team_roi_bgr, team_label="top"):
    """
    ПРОБЛЕМА: Batch обработка возвращает пустые массивы
    Результат: ['', '', '', '', ''] вместо имен игроков
    """
    try:
        # Применяем Enhanced OCR preprocessing
        from enhanced_ocr_pipeline import preprocess
        upscaled_bgr, binary = preprocess(team_roi_bgr)  # Медленная обработка!
        
        # Batch OCR с оптимизированными настройками
        data = ocr_data(binary, lang="rus", psm=6, extra_cfg=TS_WL_NAME)
        
        # Собираем текст по строкам - ЗДЕСЬ ТЕРЯЮТСЯ ДАННЫЕ
        names = [""] * len(slice_coords)
        
        for i, (text, conf, top, height) in enumerate(zip(
            data.get("text", []), data.get("conf", []), 
            data.get("top", []), data.get("height", [])
        )):
            if not text or text.strip() == "" or int(conf) < 30:  # Слишком строгий фильтр?
                continue
```

## ROI КООРДИНАТЫ (ВОЗМОЖНО НЕТОЧНЫЕ):

```python
# Текущие ROI координаты - МОГУТ БЫТЬ НЕПРАВИЛЬНЫМИ
ROI_SCORE = (0.13, 0.19, 0.47, 0.53)     # Область счета X:Y
ROI_MAP   = (0.02, 0.08, 0.35, 0.65)     # Область карты
ROI_TOP   = (0.22, 0.83, 0.05, 0.95)     # Верхняя команда
ROI_BOT   = (0.22, 0.83, 0.05, 0.95)     # Нижняя команда
NAME_CROP = (0.10, 0.90, 0.08, 0.45)     # Область имени в строке
```

## TESSERACT НАСТРОЙКИ:

```python
# Различные настройки Tesseract - КАКИЕ ОПТИМАЛЬНЫЕ?
TS_BASE = "--oem 3"
TS_FAST = "--oem 3 -c user_defined_dpi=150"  # Медленные?
TS_NO_DAWGS = "-c load_system_dawg=false -c load_freq_dawg=false"
TS_DPI = "-c user_defined_dpi=180"

# Whitelists
TS_WL_SCORE = '-c tessedit_char_whitelist=0123456789:'
TS_WL_MAP = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
```

## ТЕСТОВЫЕ РЕЗУЛЬТАТЫ:

```
=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ===
Ultra Fast OCR:
  - extract_score: 6.99 секунд (МЕДЛЕННО!)
  - extract_map: 0.39 секунд (норм)
  - Общее время: 7.39 секунд (цель: <3с)

Enhanced OCR:
  - Время: 8-35 секунд
  - Игроки: 10 найдено ✓
  - Счет: Частично работает
  - Карта: Редко работает

Batch Optimized:
  - Время: 1.3 секунды ✓
  - Игроки: 0 найдено ❌
  - Счет: None ❌
  - Карта: None ❌
```

## ВОПРОСЫ ДЛЯ CHATGPT:

1. **Почему extract_score занимает 6.99 секунд?** Один OCR вызов не должен быть таким медленным!

2. **Как оптимизировать Tesseract?** Какие настройки дадут скорость <1 секунды?

3. **ROI координаты правильные?** Может проблема в неточных областях поиска?

4. **Batch стратегия?** Как объединить несколько OCR вызовов без потери данных?

5. **Альтернативы Tesseract?** Может нужен другой OCR движок?

**ЦЕЛЬ: 100% точность + 2-5 секунд скорость для игровой платформы!**