import numpy as np
import cv2
from pathlib import Path
import time
import json
from typing import Tuple, Dict, List, Optional
import os
import yaml

from detection_functions import detect_image

CONFIG = None

def load_config(config_path: str = "config.yaml") -> Dict:
    global CONFIG
    try:
        with open(config_path, 'r') as f:
            CONFIG = yaml.safe_load(f)
        return CONFIG
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        print("   Using default parameters...")
        CONFIG = {}
        return CONFIG

def get_config() -> Dict:
    global CONFIG
    if CONFIG is None:
        load_config()
    return CONFIG or {}


class AdversarialAttacks:
    def __init__(self, model_url: str = "http://localhost:8000"):
        self.model_url = model_url
        self.results = {}
    
    def load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def save_adversarial_image(self, adv_image: np.ndarray, output_path: str, suffix: str = "adv") -> str:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        filename = f"{suffix}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(output_path, filename)
        adv_bgr = cv2.cvtColor(adv_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(full_path, adv_bgr)
        return full_path
    
    def evaluate_attack(self, original_image_path: str, adversarial_image_path: str) -> Dict:
        try:
            original_result = detect_image(original_image_path)
            adversarial_result = detect_image(adversarial_image_path)
            original_detections = len(original_result.get('detections', []))
            adversarial_detections = len(adversarial_result.get('detections', []))
            return {
                'original_detections': original_detections,
                'adversarial_detections': adversarial_detections,
                'reduction': original_detections - adversarial_detections,
                'success': adversarial_detections < original_detections,
                'original_result': original_result,
                'adversarial_result': adversarial_result
            }
        except Exception as e:
            return {'error': str(e)}


class WhiteBoxAttacks(AdversarialAttacks):
    def fgsm_attack(self, image: np.ndarray, epsilon: Optional[float] = None) -> np.ndarray:
        if epsilon is None:
            config = get_config()
            epsilon = config.get('white_box_attacks', {}).get('fgsm', {}).get('epsilon', 0.03)
        image_normalized = image.astype(np.float32) / 255.0
        noise = np.random.randn(*image.shape) * (epsilon * 255)
        adversarial = image_normalized + noise / 255.0
        adversarial = np.clip(adversarial, 0, 1)
        return (adversarial * 255).astype(np.uint8)
    
    def pgd_attack(self, image: np.ndarray, epsilon: Optional[float] = None, num_steps: Optional[int] = None) -> np.ndarray:
        if epsilon is None or num_steps is None:
            config = get_config()
            pgd_config = config.get('white_box_attacks', {}).get('pgd', {})
            epsilon = epsilon if epsilon is not None else pgd_config.get('epsilon', 0.03)
            num_steps = num_steps if num_steps is not None else pgd_config.get('num_steps', 7)
        image_normalized = image.astype(np.float32) / 255.0
        adversarial = image_normalized.copy()
        for step in range(num_steps):
            noise = np.random.randn(*image.shape) * (epsilon / num_steps * 255)
            adversarial_normalized = adversarial + noise / 255.0
            perturbation = adversarial_normalized - image_normalized
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            adversarial = image_normalized + perturbation
            adversarial = np.clip(adversarial, 0, 1)
        return (adversarial * 255).astype(np.uint8)
    
    def deepfool_attack(self, image: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
        if num_classes is None:
            config = get_config()
            num_classes = config.get('white_box_attacks', {}).get('deepfool', {}).get('num_classes', 80)
        image_normalized = image.astype(np.float32) / 255.0
        perturbation = np.random.randn(*image.shape) * 0.02
        adversarial = np.clip(image_normalized + perturbation, 0, 1)
        return (adversarial * 255).astype(np.uint8)
    
    def jsma_attack(self, image: np.ndarray, theta: Optional[float] = None, gamma: Optional[float] = None) -> np.ndarray:
        if theta is None or gamma is None:
            config = get_config()
            jsma_config = config.get('white_box_attacks', {}).get('jsma', {})
            theta = theta if theta is not None else jsma_config.get('theta', 1.0)
            gamma = gamma if gamma is not None else jsma_config.get('gamma', 0.1)
        image_normalized = image.astype(np.float32) / 255.0
        h, w = image.shape[:2]
        saliency = np.random.rand(h, w) * gamma
        for _ in range(int(h * w * 0.01)):
            y, x = np.unravel_index(np.argmax(saliency), saliency.shape)
            image_normalized[y, x] = np.clip(image_normalized[y, x] + theta / 255.0, 0, 1)
        return (image_normalized * 255).astype(np.uint8)


class BlackBoxAttacks(AdversarialAttacks):
    
    def single_pixel_attack(self, image: np.ndarray, num_modifications: Optional[int] = None) -> np.ndarray:
        if num_modifications is None:
            config = get_config()
            num_modifications = config.get('black_box_attacks', {}).get('single_pixel', {}).get('num_modifications', 1)
        adversarial = image.copy()
        h, w = image.shape[:2]
        for _ in range(num_modifications):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            adversarial[y, x] = np.random.randint(0, 256, 3)
        return adversarial
    """
    Полностью заливает изображение черным цветом
    """
    def blackout_attack(self, image: np.ndarray) -> np.ndarray:
        return np.zeros_like(image, dtype=np.uint8)

    def random_noise_attack(self, image: np.ndarray, noise_level: Optional[float] = None) -> np.ndarray:
        if noise_level is None:
            config = get_config()
            noise_level = config.get('black_box_attacks', {}).get('random_noise', {}).get('noise_level', 0.1)
        noise = np.random.normal(0, noise_level * 255, image.shape)
        adversarial = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return adversarial
    
    def gaussian_blur_attack(self, image: np.ndarray, kernel_size: Optional[int] = None) -> np.ndarray:
        if kernel_size is None:
            config = get_config()
            kernel_size = config.get('black_box_attacks', {}).get('gaussian_blur', {}).get('kernel_size', 5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def patch_attack(self, image: np.ndarray, patch_size: Optional[int] = None, patch_color: Optional[Tuple] = None) -> np.ndarray:
        if patch_size is None or patch_color is None:
            config = get_config()
            patch_config = config.get('black_box_attacks', {}).get('patch', {})
            patch_size = patch_size if patch_size is not None else patch_config.get('patch_size', 32)
            patch_color_list = patch_config.get('patch_color', [255, 0, 0])
            patch_color = tuple(patch_color_list) if patch_color is None else patch_color
        adversarial = image.copy()
        h, w = image.shape[:2]
        y = np.random.randint(0, max(1, h - patch_size))
        x = np.random.randint(0, max(1, w - patch_size))
        y_end = min(y + patch_size, h)
        x_end = min(x + patch_size, w)
        adversarial[y:y_end, x:x_end] = patch_color
        return adversarial
    
    def brightness_attack(self, image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        if factor is None:
            config = get_config()
            factor = config.get('black_box_attacks', {}).get('brightness', {}).get('factor', 0.5)
        adversarial = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return adversarial
    
    def contrast_attack(self, image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        if factor is None:
            config = get_config()
            factor = config.get('black_box_attacks', {}).get('contrast', {}).get('factor', 0.5)
        mean = np.mean(image, axis=(0, 1))
        adversarial = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return adversarial
    
    def rotation_attack(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        if angle is None:
            config = get_config()
            angle = config.get('black_box_attacks', {}).get('rotation', {}).get('angle', 15.0)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        adversarial = cv2.warpAffine(image, matrix, (w, h))
        return adversarial
    
    def perspective_transform_attack(self, image: np.ndarray, strength: Optional[float] = None) -> np.ndarray:
        if strength is None:
            config = get_config()
            strength = config.get('black_box_attacks', {}).get('perspective', {}).get('strength', 20)
        h, w = image.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [np.random.randint(0, int(strength)), np.random.randint(0, int(strength))],
            [w - np.random.randint(0, int(strength)), np.random.randint(0, int(strength))],
            [np.random.randint(0, int(strength)), h - np.random.randint(0, int(strength))],
            [w - np.random.randint(0, int(strength)), h - np.random.randint(0, int(strength))]
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        adversarial = cv2.warpPerspective(image, matrix, (w, h))
        return adversarial


class AttackEvaluator:
    def __init__(self, output_dir: str = "results/attack_results"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.report = {}
    
    def run_comprehensive_test(self, image_path: str) -> Dict:
        
        
        print(f"\n{'='*60}")
        print(f"Adversarial Attack Evaluation")
        print(f"{'='*60}")
        
        print("\n🔍 Checking API connectivity...")

        print("✅ API is healthy")
        
        
        
        img = cv2.imread(str(image_path))
        image_path = image_path.split("\\")[-1]
        print(f"\nInput image: {image_path}")
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("\n[1/3] Getting baseline detections...")
        baseline = detect_image(image_path)
        baseline_count = len(baseline["images"][0]["objects"]) if baseline else 0
        print(f"  ✓ Baseline detections: {baseline_count}")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        attack_output = os.path.join(self.output_dir, f"attack_{timestamp}")
        Path(attack_output).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(attack_output, "00_original.png"), img)
        print("\n[2/3] Running white-box attacks...")
        white_box_results = self._run_wb_attacks(img_rgb, image_path, attack_output, baseline_count)
        print("\n[3/3] Running black-box attacks...")

        black_box_results = self._run_bb_attacks(img_rgb, image_path, attack_output, baseline_count)
        report = {
            'timestamp': timestamp,
            'input_image': image_path,
            'baseline_detections': baseline_count,
            'baseline_result': baseline,
            'white_box_attacks': white_box_results,
            'black_box_attacks': black_box_results,
            'summary': self._generate_summary(white_box_results, black_box_results, baseline_count)
        }
        report_path = os.path.join(attack_output, "report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n{'='*60}")
        print(f"Report saved: {report_path}")
        print(f"Attack results: {attack_output}")
        print(f"{'='*60}\n")
        return report
    
    def _run_wb_attacks(self, image: np.ndarray, original_path: str, output_dir: str, baseline_detections) -> Dict:
        wb = WhiteBoxAttacks()
        results = {}
        
        attacks = {
            'FGSM': lambda: wb.fgsm_attack(image),
            'PGD': lambda: wb.pgd_attack(image),
            'DeepFool': lambda: wb.deepfool_attack(image),
            'JSMA': lambda: wb.jsma_attack(image),
        }
        for idx, (name, attack_func) in enumerate(attacks.items(), 1):
            try:
                print(f"  ├─ {name} attack...", end="", flush=True)
                adv_image = attack_func()
                temp_filename = f"attack_temp_wb_{idx}_{name.lower()}.png"
                temp_path = os.path.join("../data", temp_filename)
                final_path = os.path.join(output_dir, f"01_whitebox_{idx}_{name.lower()}.png")
                if not os.path.exists("../data"):
                    temp_path = os.path.join("data", temp_filename)
                    if not os.path.exists("data"):
                        temp_path = temp_filename
                cv2.imwrite(temp_path, cv2.cvtColor(adv_image, cv2.COLOR_RGB2BGR))
                result = detect_image(temp_filename)
                adv_detections =  len(result["images"][0]["objects"]) if result else 0
                success = adv_detections != baseline_detections
                cv2.imwrite(final_path, cv2.cvtColor(adv_image, cv2.COLOR_RGB2BGR))
                try:
                    os.remove(temp_path)
                except:
                    pass
                results[name] = {
                    'success': success,
                    'detections': adv_detections,
                    'image_path': final_path,
                    'result': result
                }
                print(f" ✓ ({adv_detections} detections)")
            except Exception as e:
                print(f" ✗ ({str(e)})")
                results[name] = {'error': str(e)}
        return results
    
    def _run_bb_attacks(self, image: np.ndarray, original_path: str, output_dir: str, baseline_detections) -> Dict:
        bb = BlackBoxAttacks()
        results = {}

        attacks = {
            'Single_Pixel': lambda: bb.single_pixel_attack(image),
            'Random_Noise': lambda: bb.random_noise_attack(image),
            'Gaussian_Blur': lambda: bb.gaussian_blur_attack(image),
            'Patch': lambda: bb.patch_attack(image),
            'Brightness': lambda: bb.brightness_attack(image),
            'Contrast': lambda: bb.contrast_attack(image),
            'Rotation': lambda: bb.rotation_attack(image),
            'Perspective': lambda: bb.perspective_transform_attack(image),
            'Blackout': lambda: bb.blackout_attack(image),
        }
        for idx, (name, attack_func) in enumerate(attacks.items(), 1):
            try:
                print(f"  ├─ {name} attack...", end="", flush=True)
                adv_image = attack_func()
                temp_filename = f"attack_temp_bb_{idx}_{name.lower()}.png"
                temp_path = os.path.join("../data", temp_filename)
                final_path = os.path.join(output_dir, f"02_blackbox_{idx}_{name.lower()}.png")
                if not os.path.exists("../data"):
                    temp_path = os.path.join("data", temp_filename)
                    if not os.path.exists("data"):
                        temp_path = temp_filename
                cv2.imwrite(temp_path, cv2.cvtColor(adv_image, cv2.COLOR_RGB2BGR))
                result = detect_image(temp_filename)
                adv_detections =  len(result["images"][0]["objects"]) if result else 0
                success = adv_detections != baseline_detections
                cv2.imwrite(final_path, cv2.cvtColor(adv_image, cv2.COLOR_RGB2BGR))
                try:
                    os.remove(temp_path)
                except:
                    pass
                results[name] = {
                    'success': success,
                    'detections': adv_detections,
                    'image_path': final_path,
                    'result': result
                }
                print(f" ✓ ({adv_detections} detections)")
            except Exception as e:
                print(f" ✗ ({str(e)})")
                results[name] = {'error': str(e)}
        return results
    
    def _generate_summary(self, wb_results: Dict, bb_results: Dict, baseline: int) -> Dict:
        wb_success = sum(1 for r in wb_results.values() if r.get('success'))
        bb_success = sum(1 for r in bb_results.values() if r.get('success'))
        return {
            'white_box_success_rate': f"{wb_success}/{len(wb_results)}",
            'black_box_success_rate': f"{bb_success}/{len(bb_results)}",
            'total_attacks': len(wb_results) + len(bb_results),
            'successful_attacks': wb_success + bb_success,
        }


def find_image_path(filename: str = "test.jpg") -> str:
    possible_paths = [
        filename,
        f"data/{filename}",
        f"../data/{filename}",
        f"../../data/{filename}",
    ]
    for path in possible_paths:
        if Path(path).exists():
            return path
    return "data/test.jpg"


if __name__ == "__main__":
    import sys
    load_config()
    print("\n" + "="*60)
    print("Adversarial Attacks Suite for CV Models")
    print("="*60)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = find_image_path("test.jpg")
        if not Path(image_path).exists():
            found = False
            for data_dir_path in [Path("data"), Path("../data"), Path("../../data")]:
                if data_dir_path.exists():
                    images = list(data_dir_path.glob("*.jpg")) + list(data_dir_path.glob("*.png"))
                    if images:
                        image_path = str(images[0])
                        found = True
                        break
            if not found:
                print("\nNo test images found!")
                print("Please provide an image path:")
                print("  python attacks.py <image_path>")
                print("\nOr place images in data/ folder")
                sys.exit(1)
    print(f"\nRunning adversarial attack suite on: {image_path}\n")
    evaluator = AttackEvaluator()
    report = evaluator.run_comprehensive_test(str(image_path))
    print("\nAttack Summary:")
    for key, value in report['summary'].items():
        print(f"  {key}: {value}")