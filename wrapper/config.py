"""
Управление конфигурацией и инициализацией.

Обеспечивает загрузку и доступ к параметрам конфигурации из YAML-файла.
"""

from pathlib import Path
from typing import Dict, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

# Глобальный экземпляр конфигурации
_CONFIG: Optional[Dict] = None


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Загружает конфигурацию из YAML-файла.
    
    Args:
        config_path: Путь к файлу конфигурации
    
    Returns:
        dict: Словарь конфигурации
    """
    global _CONFIG
    
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Файл конфигурации не найден: {config_path}")
            logger.info("Используются параметры по умолчанию...")
            _CONFIG = {}
            return _CONFIG
        
        with open(config_file, 'r') as f:
            _CONFIG = yaml.safe_load(f)
            logger.info(f"Конфигурация загружена из: {config_path}")
            return _CONFIG
    
    except yaml.YAMLError as e:
        logger.error(f"Ошибка парсинга YAML в {config_path}: {e}")
        _CONFIG = {}
        return _CONFIG
    
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        _CONFIG = {}
        return _CONFIG


def get_config() -> Dict:
    """
    Получает текущую конфигурацию (загружает, если ещё не загружена).
    
    Returns:
        dict: Словарь конфигурации
    """
    global _CONFIG
    
    if _CONFIG is None:
        load_config()
    
    return _CONFIG or {}


def find_image_path(filename: str = "test.jpg") -> str:
    """
    Находит путь к изображению по названию файла, ища в типичных местах.
    
    Args:
        filename: Имя файла изображения
    
    Returns:
        str: Путь к файлу изображения
    
    Raises:
        FileNotFoundError: Если изображение не найдено
    """
    from path_utils import PathManager
    
    try:
        return PathManager.find_image(filename)
    except FileNotFoundError:
        logger.error(f"Image not found: {filename}")
        # Fallback to default
        return "data/test.jpg"


def configure_logging(level: str = "INFO") -> None:
    """
    Configure logging for the module.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )