from ultralytics import YOLO
import os
import sys
import json
import datetime
import argparse

class YOLO26Detector:
    def __init__(self, model_path="yolo26x.pt", conf_thres=0.25):
        """Инициализация детектора"""
        print(f"Загрузка модели {model_path}...")
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        print(f"Модель загружена. Доступные классы: {len(self.model.names)}")
    
    def detect_folder(self, input_folder, output_folder, save_images=True, save_txt=False):
        """
        Детекция на всей папке
        Возвращает структурированный результат
        """
        os.makedirs(output_folder, exist_ok=True)
        
        results = self.model(
            source=input_folder,
            project=output_folder,
            name=".",
            save=save_images,
            save_txt=save_txt,
            exist_ok=True,
            conf=self.conf_thres
        )
        
        # Формируем детальный отчёт
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": "yolo26x",
            "conf_thres": self.conf_thres,
            "input_folder": input_folder,
            "output_folder": output_folder,
            "total_images": len(results),
            "images": []
        }
        
        for result in results:
            image_data = {
                "path": result.path,
                "filename": os.path.basename(result.path),
                "objects": []
            }
            
            for box in result.boxes:
                obj = {
                    "class": self.model.names[int(box.cls)],
                    "class_id": int(box.cls),
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist() if hasattr(box, 'xyxy') else []
                }
                image_data["objects"].append(obj)
            
            report["images"].append(image_data)
        
        return report
    
    def detect_single(self, image_path, output_folder):
        """Детекция на одном изображении"""
        os.makedirs(output_folder, exist_ok=True)
        
        results = self.model(
            source=image_path,
            project=output_folder,
            name=".",
            save=True,
            exist_ok=True,
            conf=self.conf_thres
        )
        
        return results

def main():
    """Функция для командной строки"""
    parser = argparse.ArgumentParser(description='YOLO26 детектор')
    parser.add_argument('--input', type=str, default='berzerk', help='Входная папка')
    parser.add_argument('--output', type=str, default='results', help='Выходная папка')
    parser.add_argument('--model', type=str, default='yolo26x.pt', help='Путь к модели')
    parser.add_argument('--conf', type=float, default=0.25, help='Порог уверенности')
    parser.add_argument('--json', action='store_true', help='Сохранить JSON отчёт')
    
    args = parser.parse_args()
    
    detector = YOLO26Detector(args.model, args.conf)
    report = detector.detect_folder(args.input, args.output)
    
    print(f"\nОбработано изображений: {report['total_images']}")
    
    if args.json:
        json_path = os.path.join(args.output, 'report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"JSON отчёт сохранён: {json_path}")
    
    # Вывод в консоль
    for img in report['images']:
        print(f"\n{img['filename']}: {len(img['objects'])} объектов")
        for obj in img['objects']:
            print(f"  {obj['class']} ({obj['confidence']:.2f})")

if __name__ == "__main__":
    main()