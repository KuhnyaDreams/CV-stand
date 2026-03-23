# 📖 Руководство по использованию CV-stand

Полное руководство по запуску adversarial атак на компьютерное зрение с помощью YOLOv8 детектора.

---

## 📋 Содержание

1. [Быстрый старт](#-быстрый-старт)
2. [Установка](#-установка)
3. [Структура проекта](#-структура-проекта)
4. [Типы атак](#-типы-атак)
5. [Использование](#-использование)
6. [Примеры](#-примеры)
7. [Конфигурация](#-конфигурация)
8. [Результаты](#-результаты)
9. [Отладка](#-отладка)

---

## 🚀 Быстрый старт

```bash
# 1. Сборка Docker-образов (первый раз)
docker-compose build

# 2. Запуск сервисов
docker-compose up -d

# 3. Запуск полного набора атак на test.jpg
cd wrapper
python run_full_attack.py

# 4. Просмотр результатов в ../results/attack_results/

# 5. Остановка
docker-compose down
```

---

## 🔧 Установка

### Требования
- Docker и Docker Compose
- Python 3.8+
- 4 ГБ+ свободного места на диске (для модели YOLO)
- 2 ГБ+ RAM

### Шаг 1: Клонирование и подготовка
```bash
git clone <repo-url>
cd CV-stand
mkdir -p data
```

### Шаг 2: Первичная сборка (20–30 минут)
```bash
docker-compose build
```
_Интернет требуется только на этом этапе для скачивания модели и зависимостей._

### Шаг 3: Проверка установки
```bash
# Убедитесь, что сервис запущен
docker-compose ps

# Проверьте подключение к API
curl http://localhost:8000/health
```

---

## 📁 Структура проекта

```
CV-stand/
├── core/                          # Ядро (детектор)
│   ├── app.py                     # Flask API сервис
│   ├── yolo_core.py               # Интеграция YOLO
│   ├── yolo26x.pt                 # Модель (скачивается автоматически)
│   ├── Dockerfile
│   └── requirements.txt
│
├── wrapper/                       # Обёртка (атаки)
│   ├── run_full_attack.py         # Полный набор атак (рекомендуется!)
│   ├── attack_examples.py         # Готовые примеры
│   ├── attacks.py                 # Реализация всех типов атак
│   ├── detection_functions.py     # Взаимодействие с API
│   ├── batch_processor.py         # Групповая обработка
│   ├── config.yaml                # Параметры атак
│   └── requirements.txt
│
├── data/                          # Входные изображения
├── results/                       # Выход (результаты атак)
└── docker-compose.yml
```

---

## ⚔️ Типы атак

### 🤍 Белые атаки (White-Box) — требуют знания модели

#### 1. **FGSM** (Fast Gradient Sign Method)
- Быстрая атака, основана на градиентах
- Параметр: `epsilon` (величина возмущения, по умолчанию 0.03)
- Время: < 1 сек/изображение

#### 2. **PGD** (Projected Gradient Descent)
- Итеративная атака с множественными шагами
- Параметры: `epsilon`, `num_steps` (по умолчанию 7)
- Время: 1–3 сек/изображение
- Обычно более эффективна чем FGSM

#### 3. **DeepFool**
- Минимальное возмущение для смены класса
- Параметр: `num_classes` (по умолчанию 80 для COCO)
- Время: 2–5 сек/изображение

#### 4. **JSMA** (Jacobian-based Saliency Map Attack)
- Целевая атака на пиксели с высокой чувствительностью
- Параметры: `theta` (шаг), `gamma` (коэффициент карты)
- Время: 5–10 сек/изображение

### ⬛ Чёрные атаки (Black-Box) — не требуют знания модели

#### 1. **Single Pixel Attack**
- Случайная модификация 1–N пиксел
- Быстро, но часто неэффективно
- Параметр: `num_modifications` (по умолчанию 1)

#### 2. **Random Noise**
- Добавление случайного гауссова шума
- Простая, но мощная физически реалистичная атака
- Параметр: `noise_level` (по умолчанию 0.1)

#### 3. **Gaussian Blur**
- Размытие изображения
- Часто эффективна против детекторов
- Параметр: `kernel_size` (по умолчанию 5)

#### 4. **Adversarial Patch**
- Добавление цветного прямоугольника на случайную позицию
- Параметры: `patch_size`, `patch_color`

#### 5. **Brightness**
- Изменение яркости
- Параметр: `factor` (коэффициент, 0.5 = половинная яркость)

#### 6. **Contrast**
- Изменение контраста
- Параметр: `factor`

#### 7. **Rotation**
- Поворот изображения
- Параметр: `angle` (угол в градусах, по умолчанию 15°)

#### 8. **Perspective Transform**
- Перспективное искажение
- Параметр: `strength` (величина деформации, по умолчанию 20)

---

## 💻 Использование

### Вариант 1: Полный набор атак (РЕКОМЕНДУЕТСЯ)

```bash
cd wrapper

# На изображение по умолчанию (data/test.jpg)
python run_full_attack.py

# На конкретное изображение
python run_full_attack.py /path/to/image.jpg

# Сохранить отчёт в JSON
python run_full_attack.py image.jpg --save-report report.json

# Без вывода деталей
python run_full_attack.py image.jpg --skip-report
```

**Вывод:**
```
======================================================================
🎯 Comprehensive Adversarial Attack Analysis
======================================================================

📷 Image: data/test.jpg
📁 Output: results/attack_results/

🔍 Checking API connectivity...
✅ API is healthy

[1/3] Getting baseline detections...
  ✓ Baseline detections: 3

[2/3] Running white-box attacks...
  ├─ FGSM attack... ✓ (2 detections)
  ├─ PGD attack... ✓ (1 detections)
  ├─ DeepFool attack... ✓ (2 detections)
  ├─ JSMA attack... ✓ (3 detections)

[3/3] Running black-box attacks...
  ├─ Single_Pixel attack... ✓ (3 detections)
  ├─ Random_Noise attack... ✓ (1 detections)

📊 SUMMARY REPORT
- Baseline Detection: 3 object(s)
- White-Box Success: 3/4
- Black-Box Success: 6/8
- Total Success Rate: 56.2%
```

### Вариант 2: Готовые примеры

```bash
cd wrapper

# Интерактивное меню
python attack_examples.py

# Или запустить конкретный пример
python attack_examples.py 1    # Single FGSM attack
python attack_examples.py 2    # Multiple black-box attacks
python attack_examples.py 3    # Compare attacks
python attack_examples.py 4    # Comprehensive test
python attack_examples.py 5    # Parameter scan
python attack_examples.py 6    # Sequential attack chain
python attack_examples.py 0    # Run all examples
```

### Вариант 3: Собственные скрипты

```python
from attacks import WhiteBoxAttacks, BlackBoxAttacks, AttackEvaluator
from detection_functions import detect_image
import cv2

# Загрузить изображение
image = cv2.imread("data/test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Белая атака
wb = WhiteBoxAttacks()
adversarial = wb.fgsm_attack(image_rgb, epsilon=0.05)

# Оценить эффективность
result = detect_image("data/test.jpg")
print(f"Базовых детекций: {len(result['detections'])}")

# Полная оценка
evaluator = AttackEvaluator(output_dir="results/my_attacks")
report = evaluator.run_comprehensive_test("data/test.jpg")
```

### Вариант 4: Групповая обработка

```bash
cd wrapper

# Обработать все изображения в папке
python batch_processor.py ../data
```

---

## 📚 Примеры

### Пример 1: Быстрая атака FGSM
```bash
cd wrapper
python -c "
from attacks import WhiteBoxAttacks
import cv2

img = cv2.imread('../data/test.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

wb = WhiteBoxAttacks()
adv = wb.fgsm_attack(img_rgb, epsilon=0.05)

cv2.imwrite('quick_fgsm.png', cv2.cvtColor(adv, cv2.COLOR_RGB2BGR))
print('✓ Сохранено: quick_fgsm.png')
"
```

### Пример 2: Сравнение параметров атаки
```bash
cd wrapper
python attack_examples.py 5
```
Сгенерирует FGSM атаки с разными epsilon: 0.01, 0.05, 0.1, 0.2

### Пример 3: Цепочка атак
```bash
cd wrapper
python attack_examples.py 6
```
Запустит последовательно:
1. FGSM (белая атака)
2. Random Noise (чёрная атака) на результат FGSM
3. Adversarial Patch на результат шага 2

### Пример 4: Пользовательская конфигурация
```bash
# Отредактируйте wrapper/config.yaml
nano wrapper/config.yaml

# Запустите с новыми параметрами
cd wrapper
python run_full_attack.py image.jpg
```

---

## ⚙️ Конфигурация

### config.yaml

Все параметры атак хранятся в `wrapper/config.yaml`:

```yaml
core:
  url: "http://localhost:8000"      # URL API ядра
  timeout: 30                        # Таймаут запроса (сек)

white_box_attacks:
  fgsm:
    epsilon: 0.03                   # Сила возмущения (0–1)
  pgd:
    epsilon: 0.03
    num_steps: 7                    # Количество итераций
  deepfool:
    num_classes: 80                 # Для COCO датасета
  jsma:
    theta: 1.0                      # Шаг атаки
    gamma: 0.1                      # Коэффициент карты выраженности

black_box_attacks:
  random_noise:
    noise_level: 0.1                # (0–1), 1 = добавить до 255
  gaussian_blur:
    kernel_size: 5                  # Размер ядра (нечётное)
  patch:
    patch_size: 32                  # Размер пикселя
    patch_color: [255, 0, 0]        # RGB
  rotation:
    angle: 15.0                     # Градусы
  # ... остальные параметры
```

### Как изменить параметры

1. **Для одного запуска** — отредактировать `config.yaml` перед запуском
2. **В коде** — передать параметры напрямую:

```python
wb = WhiteBoxAttacks()
adversarial = wb.fgsm_attack(image, epsilon=0.1)  # Переопределить
```

---

## 📊 Результаты

### Структура выходных данных

```
results/
├── attack_results/
│   └── attack_20260323_134025/          # Уникальная папка для каждого запуска
│       ├── 00_original.png              # Исходное изображение
│       ├── 01_whitebox_1_fgsm.png       # Результаты белых атак
│       ├── 01_whitebox_2_pgd.png
│       ├── 02_blackbox_1_single_pixel.png    # Результаты чёрных атак
│       ├── 02_blackbox_2_random_noise.png
│       └── report.json                  # Полный отчёт
│
├── singles/
│   └── 20260323_134025-test/
│       └── result.json                  # Результаты отдельно по дате
│
└── comprehensive_attacks/
    └── (результаты из attack_examples.py)
```

### Формат report.json

```json
{
  "timestamp": "20260323_134025",
  "input_image": "data/test.jpg",
  "baseline_detections": 3,
  "baseline_result": { ... },
  
  "white_box_attacks": {
    "FGSM": {
      "success": true,
      "detections": 2,
      "image_path": "...",
      "result": { "detections": [...] }
    },
    ...
  },
  
  "black_box_attacks": { ... },
  
  "summary": {
    "white_box_success_rate": "3/4",
    "black_box_success_rate": "6/8",
    "total_attacks": 12,
    "successful_attacks": 9
  }
}
```

---

## 🔍 Отладка

### Проверка статуса сервера

```bash
# Работает ли ядро?
curl http://localhost:8000/health

# Попробовать детекцию
curl -X POST http://localhost:8000/detect -F "image=@data/test.jpg"
```

### Логи сервиса

```bash
# Логи основного сервиса
docker-compose logs core

# Логи в реальном времени
docker-compose logs -f core

# Логи контейнера wrapper (если используется)
docker-compose logs wrapper
```

### Если что-то не работает

1. **API не отвечает:**
   ```bash
   docker-compose restart core
   docker-compose logs core
   ```

2. **Нет такого файла изображения:**
   ```bash
   ls -la data/
   # Поместите изображения в папку data/
   ```

3. **Ошибка при загрузке модели:**
   ```bash
   docker-compose down
   docker-compose build --no-cache
   docker-compose up
   ```

4. **Недостаточно памяти:**
   - Увеличить выделение памяти Docker
   - Или использовать более лёгкую модель (измените `yolo_core.py`)

---

## 📝 Дополнительно

### Использование собственных моделей

Отредактируйте `core/yolo_core.py`:

```python
# Строка: model = YOLO("yolov8x.pt")
model = YOLO("yolov8m.pt")  # Более лёгкая модель
# или
model = YOLO("path/to/custom/model.pt")  # Ваша модель
```

Пересоберите:
```bash
docker-compose build --no-cache
docker-compose up -d
```

### Отключение/включение конкретных атак

В `wrapper/config.yaml`:

```yaml
white_box_attacks:
  fgsm:
    enabled: true     # ← Измените на false для отключения
  pgd:
    enabled: false
```

---