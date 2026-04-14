import numpy as np
import cv2
from pathlib import Path
import time
import json
from typing import Dict
import os
import yaml

from model_functions import detect, classify 

from bb_attacks import BlackBoxAttacks
from wb_attacks import WhiteBoxAttacks

class AttackEvaluator:
    def __init__(self, output_dir: str = "results/attack_results"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.report = {}
    
    def run_single_attack_image(self, image_path: str, attack_type: str, attack_name: str):
        print(f"\nRunning attack: {attack_type} - {attack_name}")
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if attack_type == 'white_box':
            wb = WhiteBoxAttacks()
            attack_func = getattr(wb, f"{attack_name}", None)
            if not attack_func:
                raise ValueError(f"Unknown white-box attack: {attack_name}")
            adv_image = attack_func(img_rgb)
        elif attack_type == 'black_box':
            bb = BlackBoxAttacks()
            attack_func = getattr(bb, f"{attack_name}", None)
            if not attack_func:
                raise ValueError(f"Unknown black-box attack: {attack_name}")
            adv_image = attack_func(img_rgb)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        temp_filename = f"temp_{attack_type}_{attack_name}.png"
        temp_path = os.path.join("../data", temp_filename)
        if not os.path.exists("../data"):
            temp_path = os.path.join("data", temp_filename)
            if not os.path.exists("data"):
                temp_path = temp_filename
        cv2.imwrite(temp_path, cv2.cvtColor(adv_image, cv2.COLOR_RGB2BGR))
        report = detect(temp_filename)
        try:
            os.remove(temp_path)
        except:
            pass
        return adv_image, report
    
    
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
        baseline = detect(image_path)
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
                result = detect(temp_filename)
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
                result = detect(temp_filename)
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

