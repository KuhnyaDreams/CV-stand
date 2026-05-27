# Исследование возможностей открытых решений по оценке защищенности и защите систем компьютерного зрения

Этот репозиторий содержит исследовательский CV-стенд для:
- детекции объектов на изображениях;
- pose estimation на изображениях;
- анализа наличия телефона на видео;
- применения black-box и white-box атак;
- классификации атак;
- запуска защит для изображений и видео.

Проект разделен на две части:
- `core/` — API с моделями YOLO, запускается в Docker;
- `wrapper/` — локальные Python-скрипты и evaluators для атак, защит и отчетов.

## 1. Структура проекта

- [core](/B:/CV-stand/core) — FastAPI-приложение и модели YOLO.
- [wrapper](/B:/CV-stand/wrapper) — клиентские обертки, атаки, защиты, evaluators.
- [data](/B:/CV-stand/data) — входные изображения, видео и временные attacked/defended файлы.
- [results](/B:/CV-stand/results) — результаты детекции, анализа видео, отчетов по атакам и защитам.

Ключевые файлы:
- [wrapper/api/model_functions.py](/B:/CV-stand/wrapper/api/model_functions.py) — основные Python-обертки `detect`, `estimate`, `segment`, `classify`, `analyze_video_phone`.
- [wrapper/evaluation/video_attack_eval.py](/B:/CV-stand/wrapper/evaluation/video_attack_eval.py) — основной CLI для video BB атак и video defense.
- [wrapper/evaluation/attack_eval.py](/B:/CV-stand/wrapper/evaluation/attack_eval.py) — image evaluator и запуск одиночных атак с параметрами.
- [wrapper/defenses/attack_classifier.py](/B:/CV-stand/wrapper/defenses/attack_classifier.py) — классификатор типа атаки.
- [wrapper/scripts/run_all_defenses.py](/B:/CV-stand/wrapper/scripts/run_all_defenses.py) — image pipeline: атака -> классификация -> защита.

## 2. Подготовка окружения

### 2.1. Запуск `core`

Из корня проекта:

```powershell
docker-compose build
docker-compose up -d
```

После запуска:
- core API доступен на `http://localhost:8000`
- входные файлы из [data](/B:/CV-stand/data) монтируются в контейнер как `/data`
- результаты из [results](/B:/CV-stand/results) монтируются как `/results`

Остановка:

```powershell
docker-compose down
```

### 2.2. Локальное Python-окружение для `wrapper`

Из корня проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r .\wrapper\requirements.txt
```

Примечание:
- для полного white-box функционала нужен `tensorflow`;
- для большинства black-box video задач критичны `opencv-python`, `numpy`, `requests`, `matplotlib`.

## 3. Правило путей

Большинство wrapper-команд ожидают, что входные изображения и видео уже лежат в [data](/B:/CV-stand/data).

Для image API-оберток удобно передавать просто имя файла:
- `man_with_phone.jpg`
- `video3.mp4`

Для CLI из `wrapper/` обычно используются пути вида:
- `..\data\man_with_phone.jpg`
- `..\data\video3.mp4`

## 4. Простая детекция

### 4.1. Детекция на изображении

Из [wrapper](/B:/CV-stand/wrapper):

```powershell
cd B:\CV-stand\wrapper
python -c "from api.model_functions import detect; import json; print(json.dumps(detect('man_with_phone.jpg', save_images=True, show_boxes=True), ensure_ascii=False, indent=2))"
```

Что делает:
- отправляет изображение в `core`;
- сохраняет изображение с боксами и JSON-результат.

Где смотреть результат:
- [results/detection](/B:/CV-stand/results/detection)

### 4.2. Анализ видео на наличие телефона

Из [wrapper](/B:/CV-stand/wrapper):

```powershell
cd B:\CV-stand\wrapper
python -c "from api.model_functions import analyze_video_phone; import json; print(json.dumps(analyze_video_phone('video3.mp4', frame_interval=3), ensure_ascii=False, indent=2))"
```

Что делает:
- анализирует видео по сценарию `person + cell phone`;
- возвращает `detection_ratio`, `total_time_with_phone`, `avg_phone_confidence`, интервалы и другие метрики.

Где смотреть результат:
- [results/video_analysis](/B:/CV-stand/results/video_analysis)

## 5. Pose estimation

### 5.1. Pose estimation на изображении

Из [wrapper](/B:/CV-stand/wrapper):

```powershell
cd B:\CV-stand\wrapper
python -c "from api.model_functions import estimate; import json; print(json.dumps(estimate('man_with_phone.jpg', save_images=True), ensure_ascii=False, indent=2))"
```

Где смотреть результат:
- [results/estimation](/B:/CV-stand/results/estimation)

## 6. Атаки

## 6.1. Одиночная image-атака с параметрами

Самый удобный способ — через `AttackEvaluator.run_single_attack(...)`.

Пример: `random_noise_attack` с параметром `noise_level=0.18`

```powershell
cd B:\CV-stand\wrapper
python -c "from evaluation.attack_eval import AttackEvaluator; import json; ev = AttackEvaluator(output_dir='../results/attack_results'); result = ev.run_single_attack('../data/man_with_phone.jpg', 'random_noise_attack', 'black_box', {'noise_level': 0.18}, output_dir='../results/random_noise_attack_results'); print(json.dumps(result, ensure_ascii=False, indent=2))"
```

Пример: `gaussian_blur_attack` с `kernel_size=15`

```powershell
cd B:\CV-stand\wrapper
python -c "from evaluation.attack_eval import AttackEvaluator; import json; ev = AttackEvaluator(output_dir='../results/attack_results'); result = ev.run_single_attack('../data/man_with_phone.jpg', 'gaussian_blur_attack', 'black_box', {'kernel_size': 15}, output_dir='../results/gaussian_blur_attack_results'); print(json.dumps(result, ensure_ascii=False, indent=2))"
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

## 6.2. Полный image benchmark

Из [wrapper](/B:/CV-stand/wrapper):

```powershell
cd B:\CV-stand\wrapper
python -m scripts.run_full ..\data\man_with_phone.jpg
```

Этот режим прогоняет набор атак и формирует общий отчет.

## 6.3. Video BB атака через CLI

Основной entrypoint:
- [wrapper/evaluation/video_attack_eval.py](/B:/CV-stand/wrapper/evaluation/video_attack_eval.py)

Пример: `low_light_attack`

```powershell
cd B:\CV-stand\wrapper
python -m evaluation.video_attack_eval --video ..\data\video3.mp4 --attack low_light_attack --brightness-factor 0.3 --noise-level 0.14 --frame-interval 3
```

Пример: `compression_attack`

```powershell
cd B:\CV-stand\wrapper
python -m evaluation.video_attack_eval --video ..\data\video3.mp4 --attack compression_attack --jpeg-quality 15 --frame-interval 3
```

Пример: `patch_attack`

```powershell
cd B:\CV-stand\wrapper
python -m evaluation.video_attack_eval --video ..\data\video3.mp4 --attack patch_attack --patch-size 96 --patch-color 255 0 0 --patch-position fixed --patch-x 300 --patch-y 200 --patch-alpha 0.7
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

## 6.4. Параметры video-атак

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

## 6.5. Video experiments через JSON-конфиг

Пример:

```powershell
cd B:\CV-stand\wrapper
python -m evaluation.video_attack_eval --video ..\data\video3.mp4 --experiment-config "..\data\Configs\bb_all_attacks_medium.json"
```

Типовые конфиги лежат в:
- [data/Configs](/B:/CV-stand/data/Configs)

## 7. Классификация атак

## 7.1. Классификация атаки на изображении

Пример прямого вызова эвристического classifier:

```powershell
cd B:\CV-stand\wrapper
python -c "import cv2; from defenses.attack_classifier import AttackClassifier; img = cv2.imread(r'..\data\attacked_example.png'); print(AttackClassifier.classify(img, debug=True))"
```

Основной файл:
- [wrapper/defenses/attack_classifier.py](/B:/CV-stand/wrapper/defenses/attack_classifier.py)

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

## 8. Защиты

## 8.1. Защита на изображении

Image pipeline:
- применить атаку;
- классифицировать ее тип;
- применить соответствующую защиту;
- прогнать `detect(...)` на защищенном изображении.

Запуск:

```powershell
cd B:\CV-stand\wrapper
python -m scripts.run_all_defenses ..\data\man_with_phone.jpg --attack noise
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
- временные attacked/defended изображения появляются в [data](/B:/CV-stand/data)
- detect-результат сохраняется в [results/detection](/B:/CV-stand/results/detection)

## 8.2. Полный video pipeline: атака -> защита -> оценка

Пример:

```powershell
cd B:\CV-stand\wrapper
python -m evaluation.video_attack_eval --video ..\data\video3.mp4 --attack low_light_attack --brightness-factor 0.3 --noise-level 0.14 --defend --frame-interval 3 --frame-skip 2
```

Что делает:
- анализирует исходное видео;
- создает attacked video;
- классифицирует атаку по кадрам;
- применяет защиту;
- анализирует defended video;
- считает метрики восстановления.

## 8.3. Защита уже атакованного видео

Если attacked video уже готово:

```powershell
cd B:\CV-stand\wrapper
python -m evaluation.video_attack_eval --video ..\data\attack_low_light_medium_20260524_084448.mp4 --defend-attacked-video --frame-interval 3 --frame-skip 2
```

Что делает:
- принимает уже атакованное видео;
- по кадрам классифицирует тип атаки;
- применяет защиту;
- сохраняет defended video;
- считает метрики `attacked -> defended`.

Где искать результат:
- defended video — обычно в [data](/B:/CV-stand/data)
- JSON-отчет по защите — в [results/video_defense_reports](/B:/CV-stand/results/video_defense_reports)

## 8.4. Низкоуровневый video defense helper

Если нужно запустить только защиту без evaluator:

```powershell
cd B:\CV-stand\wrapper
python -c "from defenses.video_defences import defend_attacked_video; import json; result = defend_attacked_video(r'..\data\attack_low_light_medium_20260524_084448.mp4', r'..\results\defended_video.mp4', frame_skip=2); print(json.dumps(result, ensure_ascii=False, indent=2))"
```

Файл:
- [wrapper/defenses/video_defences.py](/B:/CV-stand/wrapper/defenses/video_defences.py)

## 9. Patch experiments

Для image patch sweep используется отдельный runner:
- [wrapper/evaluation/run_image_patch_experiments.py](/B:/CV-stand/wrapper/evaluation/run_image_patch_experiments.py)

Пример:

```powershell
cd B:\CV-stand\wrapper
python -m evaluation.run_image_patch_experiments --config ..\data\Configs\image_patch_sweep_v1.json
```

Где смотреть результат:
- [results/image_patch_reports](/B:/CV-stand/results/image_patch_reports)

## 10. Визуализация результатов

Доступны отдельные скрипты:
- [wrapper/evaluation/generate_attack_presentation.py](/B:/CV-stand/wrapper/evaluation/generate_attack_presentation.py)
- [wrapper/evaluation/plot_detection_ratio_drop.py](/B:/CV-stand/wrapper/evaluation/plot_detection_ratio_drop.py)
- [wrapper/evaluation/plot_attack_strength_heatmap.py](/B:/CV-stand/wrapper/evaluation/plot_attack_strength_heatmap.py)
- [wrapper/evaluation/plot_defense_effectiveness_heatmap.py](/B:/CV-stand/wrapper/evaluation/plot_defense_effectiveness_heatmap.py)

## 11. Быстрый чек-лист

1. Положить входные изображения и видео в [data](/B:/CV-stand/data)
2. Запустить `core`:

```powershell
docker-compose up -d
```

3. Перейти в [wrapper](/B:/CV-stand/wrapper)
4. Для image detection использовать `detect(...)`
5. Для image pose estimation использовать `estimate(...)`
6. Для video phone analysis использовать `analyze_video_phone(...)`
7. Для image-атак использовать `AttackEvaluator.run_single_attack(...)`
8. Для video-атак использовать `python -m evaluation.video_attack_eval ...`
9. Для image defense использовать `python -m scripts.run_all_defenses ...`
10. Для video defense использовать `--defend` или `--defend-attacked-video`

## 12. Проверка текущего состояния

На текущем состоянии проекта smoke-тесты проходят:

```powershell
cd B:\CV-stand\wrapper
python -m unittest discover -s tests -v
```

Результат:
- `17 tests`
- `OK`

