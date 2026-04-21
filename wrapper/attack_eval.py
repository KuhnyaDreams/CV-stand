"""
Adversarial attack evaluation framework.
Orchestrates attack execution, evaluates success, and generates comprehensive reports.
"""

import numpy as np
import cv2
from pathlib import Path
import time
import json
from typing import Dict, Callable, Optional, Tuple
import logging
import os
import shutil

from model_functions import detect
from bb_attacks import BlackBoxAttacks
from wb_attacks import WhiteBoxAttacks
from path_utils import PathManager

logger = logging.getLogger(__name__)


class AttackExecutor:
    """Generic attack execution engine with common logic."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize executor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
    
    def execute_attack(
        self,
        attack_func: Callable[[np.ndarray], np.ndarray],
        image: np.ndarray,
        attack_name: str,
        attack_type: str,
        temp_prefix: str = "adv_",
        save_copy_dir: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Execute single attack and get detection results.
        
        Args:
            attack_func: Attack function that takes image and returns adversarial
            image: Input image array (RGB)
            attack_name: Name of attack for logging
            attack_type: Type ('white_box' or 'black_box')
            temp_prefix: Prefix for temporary files
            
        Returns:
            Tuple of (adversarial_image, detection_result)
        """
        try:
            logger.info(f"Executing {attack_type} attack: {attack_name}")

            # Generate adversarial example
            adv_image = attack_func(image)

            # Save temporarily (to data dir) so core service can access it
            temp_filename = f"{temp_prefix}{attack_type}_{attack_name}.png"
            temp_path = PathManager.get_temp_image_path(temp_filename)

            cv2.imwrite(temp_path, cv2.cvtColor(adv_image, cv2.COLOR_RGB2BGR))

            # Optionally save a copy of adversarial image into provided directory
            if save_copy_dir:
                try:
                    PathManager.ensure_directory(save_copy_dir)
                    dst = os.path.join(save_copy_dir, Path(temp_path).name)
                    cv2.imwrite(dst, cv2.cvtColor(adv_image, cv2.COLOR_RGB2BGR))
                    logger.debug(f"Saved adversarial copy to: {dst}")
                except Exception as e:
                    logger.warning(f"Failed to save adversarial copy to {save_copy_dir}: {e}")

            # Get detection results
            result = detect(Path(temp_path).name)

            # If core returned an output folder, copy the adversarial image into it
            try:
                if result and isinstance(result, dict):
                    out_folder = result.get('output_folder') or result.get('output_path')
                    if out_folder:
                        # Try both absolute and repo-relative variants
                        candidates = [out_folder]
                        if out_folder.startswith('/'):
                            candidates.append(out_folder.lstrip('/'))

                        copied = False
                        for cand in candidates:
                            try:
                                os.makedirs(cand, exist_ok=True)
                                dst = os.path.join(cand, Path(temp_path).name)
                                shutil.copy(temp_path, dst)
                                logger.debug(f"Copied adversarial image to detection output folder: {dst}")
                                copied = True
                                break
                            except Exception:
                                continue
                        if not copied:
                            logger.warning(f"Could not copy adversarial image to any candidate output folder: {candidates}")
            except Exception as e:
                logger.debug(f"Skipping copy to output folder: {e}")

            # Cleanup temporary file from data dir
            PathManager.cleanup_file(temp_path)

            return adv_image, result

        except Exception as e:
            logger.error(f"Error executing {attack_name}: {e}")
            return None, None


class AttackEvaluator:
    """
    Main orchestrator for adversarial attack evaluation.
    Manages attack execution, result collection, and report generation.
    """
    
    # Attack definitions
    WHITE_BOX_ATTACKS = {
        'fgsm_attack': 'FGSM',
        'pgd_attack': 'PGD',
        'deepfool_attack': 'DeepFool',
        'jsma_attack': 'JSMA',
    }
    
    BLACK_BOX_ATTACKS = {
        'single_pixel_attack': 'Single_Pixel',
        'random_noise_attack': 'Random_Noise',
        'gaussian_blur_attack': 'Gaussian_Blur',
        'patch_attack': 'Patch',
        'brightness_attack': 'Brightness',
        'contrast_attack': 'Contrast',
        'rotation_attack': 'Rotation',
        'perspective_transform_attack': 'Perspective',
        'blackout_attack': 'Blackout',
    }
    
    def __init__(
        self,
        output_dir: str = "/results/attack_results",
        config: Optional[dict] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Output directory for results
            config: Optional configuration dictionary
        """
        self.output_dir = output_dir
        self.config = config or {}
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.executor = AttackExecutor(config)
        logger.info(f"Initialized AttackEvaluator with output_dir: {output_dir}")
    
    def run_comprehensive_test(self, image_path: str) -> Dict:
        """
        Run comprehensive attack evaluation on single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Complete evaluation report
        """
        logger.info(f"Starting comprehensive test on: {image_path}")
        
        print(f"\n{'='*60}")
        print(f"Adversarial Attack Evaluation")
        print(f"{'='*60}")
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_filename = Path(image_path).name
        
        print(f"\nInput image: {image_filename}")
        
        # Get baseline detections
        print("\n[1/3] Getting baseline detections...")
        baseline = detect(image_filename)
        baseline_count = self._extract_detection_count(baseline)
        print(f"  ✓ Baseline detections: {baseline_count}")
        
        # Prepare output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        attack_output = os.path.join(self.output_dir, f"attack_{timestamp}")
        PathManager.ensure_directory(attack_output)
        
        # Save original image
        cv2.imwrite(os.path.join(attack_output, "00_original.png"), img)
        
        # Run attacks
        print("\n[2/3] Running white-box attacks...")
        wb_results = self._run_attack_suite(
            img_rgb,
            self.WHITE_BOX_ATTACKS,
            WhiteBoxAttacks(self.config),
            'white_box',
            baseline_count,
            attack_output,
            '01_whitebox'
        )
        
        print("\n[3/3] Running black-box attacks...")
        bb_results = self._run_attack_suite(
            img_rgb,
            self.BLACK_BOX_ATTACKS,
            BlackBoxAttacks(self.config),
            'black_box',
            baseline_count,
            attack_output,
            '02_blackbox'
        )
        
        # Generate report
        report = {
            'timestamp': timestamp,
            'input_image': image_filename,
            'baseline_detections': baseline_count,
            'baseline_result': baseline,
            'white_box_attacks': wb_results,
            'black_box_attacks': bb_results,
            'summary': self._generate_summary(wb_results, bb_results, baseline_count)
        }
        
        # Save report
        report_path = os.path.join(attack_output, "report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"Report saved: {report_path}")
        print(f"Attack results: {attack_output}")
        print(f"{'='*60}\n")
        
        return report
    
    def _run_attack_suite(
        self,
        image: np.ndarray,
        attacks: Dict[str, str],
        attack_handler,
        attack_type: str,
        baseline_count: int,
        output_dir: str,
        output_prefix: str
    ) -> Dict:
        """
        Execute suite of attacks of same type.
        
        Args:
            image: Input image (RGB)
            attacks: Dict mapping method names to display names
            attack_handler: Attack handler instance
            attack_type: Type of attacks
            baseline_count: Number of baseline detections
            output_dir: Output directory for images
            output_prefix: Prefix for output filenames
            
        Returns:
            Dictionary of attack results
        """
        results = {}
        
        for idx, (method_name, display_name) in enumerate(attacks.items(), 1):
            try:
                print(f"  ├─ {display_name} attack...", end="", flush=True)
                
                # Get attack method
                attack_func = getattr(attack_handler, method_name, None)
                if not attack_func:
                    raise ValueError(f"Unknown attack method: {method_name}")
                
                # Execute attack
                adv_image, detection_result = self.executor.execute_attack(
                    attack_func,
                    image,
                    method_name,
                    attack_type,
                    save_copy_dir=output_dir
                )
                
                if adv_image is None:
                    raise ValueError("Attack produced None result")
                
                # Extract detection count
                adv_count = self._extract_detection_count(detection_result)
                success = adv_count != baseline_count
                
                # Save result image
                filename = f"{output_prefix}_{idx:02d}_{display_name.lower()}.png"
                final_path = os.path.join(output_dir, filename)
                cv2.imwrite(final_path, cv2.cvtColor(adv_image, cv2.COLOR_RGB2BGR))
                
                # Store results
                results[display_name] = {
                    'success': success,
                    'baseline_detections': baseline_count,
                    'adversarial_detections': adv_count,
                    'image_path': final_path,
                    'api_result': detection_result
                }
                
                print(f" ✓ ({adv_count} detections)")
                
            except Exception as e:
                logger.error(f"Error in {display_name}: {e}")
                print(f" ✗ ({str(e)})")
                results[display_name] = {'error': str(e)}
        
        return results

    def run_single_attack(
        self,
        image_path: str,
        attack_name: str,
        attack_type: str,
        params: Optional[dict] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run a single specified attack and pass parameters into it.

        Args:
            image_path: Path to the input image
            attack_name: Name of the attack method (e.g. 'fgsm_attack') or display name
            attack_type: 'white_box' or 'black_box'
            params: Optional dict of parameters to forward to the attack method
            output_dir: Optional output directory for the adversarial image

        Returns:
            Dictionary with attack result and metadata
        """
        logger.info(f"Running single attack {attack_name} ({attack_type}) on {image_path}")

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_filename = Path(image_path).name

        # Choose handler
        if attack_type == 'white_box':
            handler = WhiteBoxAttacks(self.config)
            mapping = self.WHITE_BOX_ATTACKS
        elif attack_type == 'black_box':
            handler = BlackBoxAttacks(self.config)
            mapping = self.BLACK_BOX_ATTACKS
        else:
            raise ValueError(f"Unknown attack_type: {attack_type}")

        # Resolve method name: accept either actual method name or display name
        attack_func = getattr(handler, attack_name, None)
        if not attack_func:
            # try to find method by display name
            resolved = None
            for meth, disp in mapping.items():
                if disp.lower() == attack_name.lower() or meth.lower() == attack_name.lower():
                    resolved = meth
                    break
            if resolved:
                attack_func = getattr(handler, resolved, None)

        if not attack_func:
            raise ValueError(f"Attack method not found: {attack_name}")

        params = params or {}

        # Wrap attack to accept single-image signature required by executor
        def attack_wrapper(img_arr: np.ndarray) -> np.ndarray:
            return attack_func(img_arr, **params)

        # Prepare output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_out = output_dir or os.path.join(self.output_dir, f"single_attack_{timestamp}")
        PathManager.ensure_directory(base_out)

        # Execute
        adv_image, detection_result = self.executor.execute_attack(
            attack_wrapper,
            img_rgb,
            attack_name,
            attack_type,
            temp_prefix="single_",
            save_copy_dir=base_out
        )

        if adv_image is None:
            return {'error': 'Attack produced no adversarial image'}

        adv_count = self._extract_detection_count(detection_result)
        baseline = self._extract_detection_count(detect(image_filename))
        success = adv_count != baseline

        # Save adversarial image
        '''filename = f"single_{attack_type}_{attack_name}.png"
        final_path = os.path.join(base_out, filename)
        cv2.imwrite(final_path, cv2.cvtColor(adv_image, cv2.COLOR_RGB2BGR)) '''

        result = {
            'attack': attack_name,
            'attack_type': attack_type,
            'params': params,
            'input_image': image_filename,
            #'output_image': final_path,
            'baseline_detections': baseline,
            'adversarial_detections': adv_count,
            'success': success,
            'api_result': detection_result
        }

        return result
    
    @staticmethod
    def _extract_detection_count(result: Optional[Dict]) -> int:
        """
        Safely extract detection count from API result.
        
        Args:
            result: Result from API call
            
        Returns:
            Number of detections (0 if error)
        """
        try:
            if not result or 'images' not in result:
                logger.warning("Invalid result format")
                return 0
            
            images = result.get('images', [])
            if not images:
                return 0
            
            first_image = images[0]
            objects = first_image.get('objects', [])
            return len(objects)
            
        except (KeyError, TypeError, IndexError) as e:
            logger.error(f"Error extracting detection count: {e}")
            return 0
    
    @staticmethod
    def _generate_summary(
        wb_results: Dict,
        bb_results: Dict,
        baseline: int
    ) -> Dict:
        """
        Generate evaluation summary statistics.
        
        Args:
            wb_results: White-box attack results
            bb_results: Black-box attack results
            baseline: Baseline detection count
            
        Returns:
            Summary dictionary
        """
        wb_successful = sum(
            1 for r in wb_results.values()
            if isinstance(r, dict) and r.get('success')
        )
        bb_successful = sum(
            1 for r in bb_results.values()
            if isinstance(r, dict) and r.get('success')
        )
        
        total_attacks = len(wb_results) + len(bb_results)
        total_successful = wb_successful + bb_successful
        
        return {
            'baseline_detections': baseline,
            'white_box_attacks': {
                'total': len(wb_results),
                'successful': wb_successful,
                'success_rate': f"{wb_successful}/{len(wb_results)}"
            },
            'black_box_attacks': {
                'total': len(bb_results),
                'successful': bb_successful,
                'success_rate': f"{bb_successful}/{len(bb_results)}"
            },
            'overall': {
                'total_attacks': total_attacks,
                'successful_attacks': total_successful,
                'success_rate': f"{total_successful}/{total_attacks}",
                'success_percentage': f"{100*total_successful/total_attacks:.1f}%" if total_attacks > 0 else "0%"
            }
        }

