
# Исследование возможностей открытых решений по оценке защищенности и защите систем компьютерного зрения

## Самое важное

1. Положить входные изображения и видео в `data/`
2. Сбилдить и запустить стенд:

```
docker-compose up -d --build
```

3. Перейти в `wrapper/`
4. Для image detection использовать `detect(...)`
5. Для image pose estimation использовать `estimate(...)`
6. Для video phone analysis использовать `analyze_video_phone(...)`
7. Для image-атак использовать `AttackEvaluator.run_single_attack(...)`
8. Для video-атак использовать `python -m evaluation.video_attack_eval ...`
9. Для image defense использовать `python -m scripts.run_all_defenses ...`
10. Для video defense использовать `--defend` или `--defend-attacked-video`

## 1. О проекте

Этот репозиторий содержит исследовательский CV-стенд для:
- детекции объектов на изображениях;
- pose estimation на изображениях;
- анализа наличия телефона на видео;
- применения black-box и white-box атак;
- классификации атак;
- запуска защит для изображений и видео.


###  Структура проекта

- `core/` — FastAPI-приложение и модели YOLO, запускается в Docker.
- `wrapper/` — клиентские обертки, атаки, защиты, evaluators.
- `data/` — входные изображения, видео и временные attacked/defended файлы.
- `results/` — результаты детекции, анализа видео, отчетов по атакам и защитам.

Ключевые файлы:
- wrapper/api/model_functions.py — основные Python-обертки `detect`, `estimate`, `segment`, `classify`, `analyze_video_phone`.
- wrapper/evaluation/video_attack_eval.py — основной CLI для video BB атак и video defense.
- wrapper/evaluation/attack_eval.py — image evaluator и запуск одиночных атак с параметрами.
- wrapper/defenses/attack_classifier.py — классификатор типа атаки.
- wrapper/scripts/run_all_defenses.py — image pipeline: атака -> классификация -> защита.

## 2. Подготовка окружения

### 2.1. Запуск `core`

Из корня проекта:

```
docker compose build
docker compose up -d
```

После запуска:
- core API доступен на `http://localhost:8000`
- Документация API доступен на `http://localhost:8000/docs`
- входные файлы из data монтируются в контейнер как `/data`
- результаты из results монтируются как `/results`

Остановка:

```
docker compose down
```

### 2.2. Локальное Python-окружение для `wrapper`

Из корня проекта:

#### Windows
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r .\wrapper\requirements.txt
```

#### Linux/MacOS
```
python -m venv .venv
source .venv/bin/activate
pip install -r ./wrapper/requirements.txt
```

Примечание:
- для полного white-box функционала нужен `tensorflow`;
- для большинства black-box video задач критичны `opencv-python`, `numpy`, `requests`, `matplotlib`.

## 3. Правило путей

Большинство wrapper-команд ожидают, что входные изображения и видео уже лежат в data

Для image API-оберток удобно передавать просто имя файла:
- `man_with_phone.jpg`
- `video3.mp4`

Для CLI из `wrapper/` обычно используются пути вида:
- `../data/man_with_phone.jpg`
- `../data/video3.mp4`

## 4. Полный video pipeline: атака -> защита -> оценка

Пример:

```
python -m evaluation.video_attack_eval \
--video ../data/video3.mp4 \
--attack low_light_attack \
--brightness-factor 0.3 \
--noise-level 0.14 \
--defend \
--frame-interval 3 \
--frame-skip 2
```

Что делает:
- анализирует исходное видео;
- создает attacked video;
- классифицирует атаку по кадрам;
- применяет защиту;
- анализирует defended video;
- считает метрики восстановления.

## 5. Простая детекция

### 5.1. Детекция на изображении


```
python -c "from api.model_functions import detect; import json;
print(json.dumps(detect('man_with_phone.jpg', save_images=True, show_boxes=True),
ensure_ascii=False, indent=2))"
```

Что делает:
- отправляет изображение в `core`;
- сохраняет изображение с боксами и JSON-результат.

Где смотреть результат:
- results/detection

### 5.2. Анализ видео на наличие телефона


```
python -c "from api.model_functions import analyze_video_phone; import json;
print(json.dumps(analyze_video_phone('video3.mp4', frame_interval=3),
ensure_ascii=False, indent=2))"
```

Что делает:
- анализирует видео по сценарию `person + cell phone`;
- возвращает `detection_ratio`, `total_time_with_phone`, `avg_phone_confidence`, интервалы и другие метрики.

Где смотреть результат:
- results/video_analysis

## 6. Pose estimation

### 6.1. Pose estimation на изображении

Из wrapper

```
python -c "from api.model_functions import estimate; import json;
print(json.dumps(estimate('man_with_phone.jpg', save_images=True),
ensure_ascii=False, indent=2))"
```

Где смотреть результат:
- results/estimation

## 7. Атаки

## 7.1. Одиночная image-атака с параметрами

Самый удобный способ — через `AttackEvaluator.run_single_attack(...)`.

Пример: `random_noise_attack` с параметром `noise_level=0.18`

```
python -c "from evaluation.attack_eval import AttackEvaluator;
import json; ev = AttackEvaluator(output_dir='../results/attack_results');
result = ev.run_single_attack('../data/man_with_phone.jpg', 'random_noise_attack',
'black_box', {'noise_level': 0.18}, output_dir='../results/random_noise_attack_results');
print(json.dumps(result, ensure_ascii=False, indent=2))"
```

Пример: `gaussian_blur_attack` с `kernel_size=15`

```
python -c "from evaluation.attack_eval import AttackEvaluator;
import json; ev = AttackEvaluator(output_dir='../results/attack_results');
result = ev.run_single_attack('../data/man_with_phone.jpg', 'gaussian_blur_attack',
'black_box', {'kernel_size': 15}, output_dir='../results/gaussian_blur_attack_results');
print(json.dumps(result, ensure_ascii=False, indent=2))"
```

Поддерживаемые основные image BB атаки:
- `single_pixel_attack`
- `random_noise_attack`
- `gaussian_blur_attack`
- `patch_attack`
- `brightness_attack`
- `contrast_attack`
- `rotation_attack`
- `perspective_transform_attack`
- `blackout_attack`

## 7.2. Полный image benchmark

Из wrapper

```
python -m scripts.run_full ../data/man_with_phone.jpg
```

Этот режим прогоняет набор атак и формирует общий отчет.

## 7.3. Video BB атака через CLI

Основной entrypoint:
- wrapper/evaluation/video_attack_eval.py

Пример: `low_light_attack`

```
python -m evaluation.video_attack_eval \
--video ../data/video3.mp4 \
--attack low_light_attack \
--brightness-factor 0.3 \
--noise-level 0.14 \
--frame-interval 3
```

Пример: `compression_attack`

```
python -m evaluation.video_attack_eval \
--video ../data/video3.mp4 \
--attack compression_attack \
--jpeg-quality 15 \
--frame-interval 3
```

Пример: `patch_attack`

```
python -m evaluation.video_attack_eval \
--video ../data/video3.mp4 \
--attack patch_attack \
--patch-size 96 \
--patch-color 255 0 0 \
--patch-position fixed \
--patch-x 300 \
--patch-y 200 \
--patch-alpha 0.7
```

Поддерживаемые video BB атаки:
- `gaussian_blur_attack`
- `motion_blur_attack`
- `random_noise_attack`
- `low_light_attack`
- `brightness_attack`
- `contrast_attack`
- `compression_attack`
- `downscale_upscale_attack`
- `frame_drop_attack`
- `blackout_attack`
- `patch_attack`

## 7.4. Параметры video-атак

Основные CLI-флаги:
- `--kernel-size` — для `gaussian_blur_attack`
- `--motion-angle` — для `motion_blur_attack`
- `--noise-level` — для `random_noise_attack`
- `--brightness-factor` — для `brightness_attack` и `low_light_attack`
- `--contrast-factor` — для `contrast_attack`
- `--jpeg-quality` — для `compression_attack`
- `--scale-factor` — для `downscale_upscale_attack`
- `--drop-every-n` — для `frame_drop_attack`
- `--patch-size`, `--patch-color`, `--patch-position`, `--patch-x`, `--patch-y`, `--patch-alpha`, `--patch-texture`, `--texture-strength`, `--edge-softness`, `--patch-shape` — для `patch_attack`

Временное применение атаки:
- `--temporal-mode always`
- `--temporal-mode flicker`
- `--flicker-period`
- `--flicker-active-ratio`

## 7.5. Video experiments через JSON-конфиг

Пример:

```
python -m evaluation.video_attack_eval \
--video ../data/video3.mp4 \
--experiment-config "../data/Configs/bb_all_attacks_medium.json"
```

Типовые конфиги лежат в:
- data/Configs

## 8. Классификация атак

## 8.1. Классификация атаки на изображении

Пример прямого вызова эвристического classifier:

```
python -c "import cv2; from defenses.attack_classifier import AttackClassifier;
img = cv2.imread(r'../data/attacked_example.png');
print(AttackClassifier.classify(img, debug=True))"
```

Основной файл:
- wrapper/defenses/attack_classifier.py

Классификатор умеет распознавать, в том числе:
- `gaussian_blur`
- `motion_blur`
- `random_noise`
- `low_light`
- `brightness`
- `contrast`
- `compression`
- `downscale_upscale`
- `frame_drop`
- `blackout`
- `patch`

## 9. Защиты

## 9.1. Защита на изображении

Image pipeline:
- применить атаку;
- классифицировать ее тип;
- применить соответствующую защиту;
- прогнать `detect(...)` на защищенном изображении.

Запуск:

```
cd wrapper
python -m scripts.run_all_defenses ../data/man_with_phone.jpg --attack noise
```

Другие варианты `--attack`:
- `blackout`
- `blur`
- `brightness`
- `contrast`
- `noise`
- `patch`
- `perspective`
- `rotation`
- `single_pixel`

Где искать артефакты:
- временные attacked/defended изображения появляются в data
- detect-результат сохраняется в results/detection

## 9.2. Защита уже атакованного видео

Если attacked video уже готово:

```
python -m evaluation.video_attack_eval \
--video ../data/attack_low_light_medium_20260524_084448.mp4 \
--defend-attacked-video \
--frame-interval 3 \
--frame-skip 2
```

Что делает:
- принимает уже атакованное видео;
- по кадрам классифицирует тип атаки;
- применяет защиту;
- сохраняет defended video;
- считает метрики `attacked -> defended`.

Где искать результат:
- defended video — обычно в data
- JSON-отчет по защите — в results/video_defense_reports

## 9.3. Низкоуровневый video defense helper

Если нужно запустить только защиту без evaluator:

```
cd wrapper
python -c "from defenses.video_defences import defend_attacked_video; import json;
result = defend_attacked_video(r'../data/attack_low_light_medium_20260524_084448.mp4',
r'../results/defended_video.mp4', frame_skip=2);
print(json.dumps(result, ensure_ascii=False, indent=2))"
```

Файл:
- wrapper/defenses/video_defences.py

## 10. Patch experiments

Для image patch sweep используется отдельный runner:
- wrapper/evaluation/run_image_patch_experiments.py

Пример:

```
python -m evaluation.run_image_patch_experiments \
--config ../data/Configs/image_patch_sweep_v1.json
```

Где смотреть результат:
- results/image_patch_reports

## 11. Визуализация результатов

Доступны отдельные скрипты:
- wrapper/evaluation/generate_attack_presentation.py
- wrapper/evaluation/plot_detection_ratio_drop.py
- wrapper/evaluation/plot_attack_strength_heatmap.py
- wrapper/evaluation/plot_defense_effectiveness_heatmap.py





