#!/usr/bin/env python3
"""
Автоматический запуск тестов качества OCR.
Можно добавить в cron для регулярных проверок.
"""

import os
import sys
import json
import time
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from golden_test_runner import main as run_golden_test

def load_config():
    """Загружает конфигурацию мониторинга."""
    config_path = Path(__file__).parent / "monitoring_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_alerts(summary_path):
    """Проверяет результаты и генерирует алерты при необходимости."""
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    config = load_config()
    targets = config["monitoring"]["performance_targets"]
    
    alerts = []
    stats = summary["stats"]
    
    # Проверка времени обработки
    if stats.get("average_time", 0) > targets["max_processing_time"]:
        alerts.append(f"⚠️ Среднее время обработки превышает цель: {stats['average_time']:.2f}с > {targets['max_processing_time']}с")
    
    # Проверка успешности
    if stats.get("success_rate", 0) < targets["min_success_rate"]:
        alerts.append(f"⚠️ Уровень успешности ниже цели: {stats['success_rate']:.1%} < {targets['min_success_rate']:.1%}")
    
    # Проверка точности
    if stats.get("average_quality", 0) < targets["min_accuracy"]:
        alerts.append(f"⚠️ Средняя точность ниже цели: {stats['average_quality']:.1%} < {targets['min_accuracy']:.1%}")
    
    # Проверка достижения цели ChatGPT-5
    if stats.get("chatgpt5_achievement_rate", 0) < targets["chatgpt5_target_achievement"]:
        alerts.append(f"⚠️ Цель ChatGPT-5 достигается недостаточно часто: {stats['chatgpt5_achievement_rate']:.1%} < {targets['chatgpt5_target_achievement']:.1%}")
    
    if alerts:
        alert_file = Path(__file__).parent / "alerts" / f"alert_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(alert_file, "w", encoding="utf-8") as f:
            f.write("🚨 МОНИТОРИНГ OCR: ОБНАРУЖЕНЫ ПРОБЛЕМЫ\n\n")
            f.write("\n".join(alerts))
            f.write(f"\n\nВремя проверки: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🚨 Сгенерирован алерт: {alert_file}")
        
        for alert in alerts:
            print(alert)
    else:
        print("✅ Все показатели в норме")

def main():
    """Главная функция автоматического тестирования."""
    
    print(f"🔄 Автоматическая проверка качества OCR - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Переходим в директорию проекта
    os.chdir(Path(__file__).parent.parent)
    
    # Запускаем тесты
    try:
        run_golden_test()
        
        # Ищем последний сгенерированный отчет
        import glob
        summaries = glob.glob("golden_test_summary_*.json")
        if summaries:
            latest_summary = max(summaries, key=os.path.getctime)
            check_alerts(latest_summary)
            
            # Перемещаем отчеты в папку reports
            reports_dir = Path("golden/reports")
            for file_pattern in ["golden_test_results_*.csv", "golden_test_summary_*.json"]:
                for file_path in glob.glob(file_pattern):
                    shutil.move(file_path, reports_dir / os.path.basename(file_path))
                    
        print("✅ Автоматическая проверка завершена")
        
    except Exception as e:
        error_file = Path("golden/alerts") / f"error_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"❌ Ошибка автоматической проверки OCR:\n\n{str(e)}")
        print(f"❌ Ошибка: {e}")
        raise

if __name__ == "__main__":
    main()
