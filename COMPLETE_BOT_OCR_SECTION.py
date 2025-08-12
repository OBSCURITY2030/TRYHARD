#!/usr/bin/env python3
"""
ПОЛНЫЙ КОД СЕКЦИИ OCR ИЗ TELEGRAM БОТА
Для анализа ChatGPT - где именно проблема и как исправить
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# === OCR СИСТЕМЫ ИМПОРТЫ ===
try:
    from ultra_fast_ocr import process_screenshot_ultra_fast
    ULTRA_FAST_AVAILABLE = True
    print("⚡ УЛЬТРА-БЫСТРАЯ OCR система загружена - 3 OCR вызова = ВСЯ информация")
except ImportError as e:
    ULTRA_FAST_AVAILABLE = False
    print(f"⚠️ Ультра-быстрая система недоступна: {e}")

try:
    from fixed_batch_optimized_pipeline import process_screenshot_fixed
    FIXED_BATCH_AVAILABLE = True
    print("🔧 ИСПРАВЛЕННАЯ Batch OCR система загружена - быстро + полные данные")
except ImportError as e:
    FIXED_BATCH_AVAILABLE = False
    print(f"⚠️ Исправленная система недоступна: {e}")

try:
    from batch_optimized_pipeline import process_screenshot_batch
    BATCH_PIPELINE_AVAILABLE = True
    print("🏆 Batch-оптимизированный OCR пайплайн загружен (ChatGPT-5 Final)")
except ImportError as e:
    BATCH_PIPELINE_AVAILABLE = False
    print(f"⚠️ Batch пайплайн недоступен: {e}")

try:
    from enhanced_ocr_pipeline import process_screenshot
    ENHANCED_PIPELINE_AVAILABLE = True
    print("✅ Enhanced OCR Pipeline активирован")
except ImportError as e:
    ENHANCED_PIPELINE_AVAILABLE = False
    print(f"⚠️ Enhanced пайплайн недоступен: {e}")

try:
    from production_tesseract_ocr import ProductionOCR
    production_ocr = ProductionOCR()
    PRODUCTION_OCR_AVAILABLE = True
    print("✅ Production Tesseract OCR активирован")
except ImportError as e:
    PRODUCTION_OCR_AVAILABLE = False
    print(f"⚠️ Production OCR недоступен: {e}")

logger = logging.getLogger(__name__)

# === ВЫБОР ЛУЧШЕЙ OCR СИСТЕМЫ ===
def get_best_ocr_function():
    """Выбор лучшего доступного OCR пайплайна."""
    if ULTRA_FAST_AVAILABLE:
        return process_screenshot_ultra_fast, "⚡ УЛЬТРА-БЫСТРАЯ: 3 OCR вызова (1-3s)"
    elif FIXED_BATCH_AVAILABLE:
        return process_screenshot_fixed, "ИСПРАВЛЕНА: Быстро + Полные данные (2-5s)"
    elif BATCH_PIPELINE_AVAILABLE:
        return process_screenshot_batch, "Batch Pipeline (2-5s target)"
    elif ENHANCED_PIPELINE_AVAILABLE:
        return process_screenshot, "Enhanced Pipeline (8-10s)"
    elif PRODUCTION_OCR_AVAILABLE:
        return production_ocr.process_screenshot, "Production Pipeline (15-25s)"
    else:
        return None, "NO OCR AVAILABLE"

# Выбираем лучшую систему при старте
ocr_function, method_name = get_best_ocr_function()
print(f"📊 Используется: {method_name}")

# === ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ СКРИНШОТА ===
@dp.message_handler(content_types=['photo'])
async def handle_screenshot(message: types.Message):
    """
    ГЛАВНАЯ ФУНКЦИЯ - ОБРАБОТКА СКРИНШОТА STANDOFF 2
    ТУТ ПРОИСХОДИТ ОСНОВНАЯ МАГИЯ OCR
    """
    user_id = str(message.from_user.id)
    
    try:
        # Проверяем регистрацию
        if user_id not in users:
            await message.answer(
                "❌ Сначала зарегистрируйтесь!\n"
                "Используйте /start для регистрации"
            )
            return
        
        # Скачиваем фото
        photo = message.photo[-1]
        file_info = await bot.get_file(photo.file_id)
        file_bytes = await bot.download_file(file_info.file_path)
        
        # Сообщение о начале обработки
        processing_msg = await message.answer(
            f"🔍 Обрабатываю скриншот...\n"
            f"📊 Метод: {method_name}\n"
            f"⏱️ Ожидаемое время: ~3-8 секунд"
        )
        
        # Засекаем время
        start_time = time.time()
        
        # Получаем список известных игроков для сопоставления
        known_players = []
        for uid, user_data in users.items():
            if isinstance(user_data, dict):
                # Добавляем Standoff 2 ID
                if user_data.get('standoff2_id'):
                    known_players.append(user_data['standoff2_id'])
                # Добавляем игровое имя
                if user_data.get('game_name'):
                    known_players.append(user_data['game_name'])
        
        logger.info(f"Screenshot processing started by {user_id}, known players: {len(known_players)}")
        
        # === ОСНОВНАЯ OCR ОБРАБОТКА - ЗДЕСЬ ПРОБЛЕМА! ===
        if ocr_function:
            # Читаем байты изображения
            image_bytes = file_bytes.read()
            
            # ВЫЗЫВАЕМ OCR СИСТЕМУ
            result = ocr_function(image_bytes, known_players)
            
            processing_time = time.time() - start_time
            
            # Логируем результат
            logger.info(f"OCR completed in {processing_time:.2f}s: success={result.get('success')}")
            
            if result.get('success'):
                await process_ocr_result(message, result, processing_msg, processing_time)
            else:
                error_msg = result.get('error', 'Неизвестная ошибка')
                await processing_msg.edit_text(
                    f"❌ Ошибка обработки скриншота\n"
                    f"Причина: {error_msg}\n"
                    f"Время: {processing_time:.1f}с"
                )
        else:
            await processing_msg.edit_text("❌ OCR система недоступна")
            
    except Exception as e:
        logger.error(f"Screenshot processing error: {e}")
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        await message.answer(
            f"❌ Произошла ошибка при обработке\n"
            f"Время: {processing_time:.1f}с\n"
            f"Ошибка: {str(e)}"
        )

async def process_ocr_result(message: types.Message, result: Dict, processing_msg, processing_time: float):
    """
    ОБРАБОТКА РЕЗУЛЬТАТА OCR - ПОКАЗЫВАЕТ ЧТО ИМЕННО НЕ РАБОТАЕТ
    """
    try:
        # Извлекаем данные
        score = result.get('score')
        map_name = result.get('map') 
        players = result.get('players', [])
        winner_side = result.get('winner_side', 'unknown')
        
        # Анализируем качество данных
        has_score = bool(score and score != 'None')
        has_map = bool(map_name and map_name != 'None')
        has_players = len(players) >= 8  # Минимум 8 игроков
        
        # Формируем отчет
        quality_score = 0
        if has_score:
            quality_score += 30
        if has_map:
            quality_score += 30
        if has_players:
            quality_score += 40
        
        # Создаем сообщение с результатами
        result_text = f"📊 **РЕЗУЛЬТАТ OCR ОБРАБОТКИ**\n\n"
        result_text += f"⏱️ Время: {processing_time:.2f}с\n"
        result_text += f"🎯 Качество: {quality_score}%\n\n"
        
        # Счет
        if has_score:
            result_text += f"🏆 Счет: **{score}** (Победитель: {winner_side})\n"
        else:
            result_text += f"❌ Счет: Не найден\n"
        
        # Карта
        if has_map:
            result_text += f"🗺️ Карта: **{map_name}**\n"
        else:
            result_text += f"❌ Карта: Не найдена\n"
        
        # Игроки
        result_text += f"👥 Игроки: {len(players)}/10\n\n"
        
        if players:
            result_text += "**НАЙДЕННЫЕ ИГРОКИ:**\n"
            
            # Группируем по командам
            top_players = [p for p in players if p.get('team') == 'top']
            bottom_players = [p for p in players if p.get('team') == 'bottom']
            
            result_text += "🔵 **Верхняя команда:**\n"
            for i, player in enumerate(top_players[:5], 1):
                name = player.get('raw', 'Unknown')[:20]
                matched = player.get('matched')
                if matched:
                    result_text += f"{i}. {name} → **{matched}**\n"
                else:
                    result_text += f"{i}. {name}\n"
            
            result_text += "\n🔴 **Нижняя команда:**\n"
            for i, player in enumerate(bottom_players[:5], 1):
                name = player.get('raw', 'Unknown')[:20]
                matched = player.get('matched')
                if matched:
                    result_text += f"{i}. {name} → **{matched}**\n"
                else:
                    result_text += f"{i}. {name}\n"
        else:
            result_text += "❌ **ИГРОКИ НЕ НАЙДЕНЫ**\n"
        
        # Диагностика
        result_text += f"\n🔧 **ДИАГНОСТИКА:**\n"
        result_text += f"Метод: {method_name}\n"
        
        if quality_score < 70:
            result_text += "⚠️ **НИЗКОЕ КАЧЕСТВО РАСПОЗНАВАНИЯ**\n"
            if not has_score:
                result_text += "- Счет не найден\n"
            if not has_map:
                result_text += "- Карта не найдена\n"
            if not has_players:
                result_text += f"- Мало игроков ({len(players)}/10)\n"
        
        # Обновляем сообщение
        await processing_msg.edit_text(result_text, parse_mode='Markdown')
        
        # Если данные неполные, предлагаем действия
        if quality_score < 70:
            keyboard = InlineKeyboardMarkup()
            keyboard.add(InlineKeyboardButton("🔄 Попробовать снова", callback_data=f"retry_ocr_{message.message_id}"))
            keyboard.add(InlineKeyboardButton("📞 Сообщить о проблеме", callback_data="report_problem"))
            
            await message.answer(
                "⚠️ Качество распознавания низкое. Что делать?",
                reply_markup=keyboard
            )
    
    except Exception as e:
        logger.error(f"Error processing OCR result: {e}")
        await processing_msg.edit_text(
            f"❌ Ошибка при обработке результата OCR\n"
            f"Время: {processing_time:.2f}с\n"
            f"Ошибка: {str(e)}"
        )

# === ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ===
@dp.callback_query_handler(lambda c: c.data.startswith('retry_ocr_'))
async def retry_ocr_processing(callback_query: types.CallbackQuery):
    """Повторная обработка OCR"""
    await callback_query.answer("🔄 Функция в разработке")

@dp.callback_query_handler(lambda c: c.data == 'report_problem')
async def report_ocr_problem(callback_query: types.CallbackQuery):
    """Сообщить о проблеме OCR"""
    await callback_query.answer("📞 Проблема зафиксирована")

if __name__ == "__main__":
    print("🎯 OCR секция Telegram бота загружена")
    print(f"📊 Активная система: {method_name}")
    print("⚠️ ПРОБЛЕМЫ:")
    print("1. OCR занимает 7-35 секунд вместо 2-5")
    print("2. Не находит счет и карту")
    print("3. Нужна помощь ChatGPT для оптимизации!")