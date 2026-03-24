from attacks import WhiteBoxAttacks, BlackBoxAttacks, AttackEvaluator, load_config
import cv2
import numpy as np
from pathlib import Path
import os


def find_image(filename: str = "test.jpg") -> str:
    possible_paths = [
        filename,
        f"data/{filename}",
        f"../data/{filename}",
        f"../../data/{filename}",
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    return possible_paths[1]


def example_1_single_attack():
    print("Example 1: Single FGSM Attack")
    print("-" * 40)
    image_path = find_image("test.jpg")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        print("\nMake sure test.jpg exists in data/ folder")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    wb = WhiteBoxAttacks()
    adversarial = wb.fgsm_attack(image_rgb, epsilon=0.05)
    output_path = "results/example1_fgsm_attack.png"
    Path("results").mkdir(exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(adversarial, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved to: {output_path}\n")


def example_2_black_box_attacks():
    print("Example 2: Multiple Black-Box Attacks")
    print("-" * 40)
    image_path = find_image("test.jpg")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        print("\nMake sure test.jpg exists in data/ folder")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bb = BlackBoxAttacks()
    attacks = {
        "patch": bb.patch_attack(image_rgb, patch_size=50),
        "noise": bb.random_noise_attack(image_rgb, noise_level=0.15),
        "blur": bb.gaussian_blur_attack(image_rgb, kernel_size=7),
    }
    Path("results").mkdir(exist_ok=True)
    for name, adversarial in attacks.items():
        output_path = f"results/example2_{name}_attack.png"
        cv2.imwrite(output_path, cv2.cvtColor(adversarial, cv2.COLOR_RGB2BGR))
        print(f"✓ {name.capitalize():10} > {output_path}")
    print()


def example_3_compare_attacks():
    print("Example 3: Attack Comparison")
    print("-" * 40)
    image_path = find_image("test.jpg")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        print("\nMake sure test.jpg exists in data/ folder")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    wb = WhiteBoxAttacks()
    bb = BlackBoxAttacks()
    print("White-box attacks:")
    wb_attacks = {
        "FGSM": wb.fgsm_attack(image_rgb),
        "PGD": wb.pgd_attack(image_rgb),
        "DeepFool": wb.deepfool_attack(image_rgb),
    }
    for name in wb_attacks:
        print(f"  ✓ {name}")
    print("\nBlack-box attacks:")
    bb_attacks = {
        "Patch": bb.patch_attack(image_rgb),
        "Rotation": bb.rotation_attack(image_rgb, angle=20),
        "Perspective": bb.perspective_transform_attack(image_rgb),
    }
    for name in bb_attacks:
        print(f"  ✓ {name}")
    print()


def example_4_comprehensive_test():
    print("Example 4: Comprehensive Attack Evaluation")
    print("-" * 40)
    print("Running all attacks on an image...\n")
    image_path = find_image("test.jpg")
    if not Path(image_path).exists():
        print(f"⚠ Image not found: {image_path}")
        print("Using first available image from data folder...\n")
        for data_dir_path in [Path("data"), Path("../data"), Path("../../data")]:
            if data_dir_path.exists():
                images = list(data_dir_path.glob("*.jpg")) + list(data_dir_path.glob("*.png"))
                if images:
                    image_path = str(images[0])
                    break
        else:
            print("No images found in data/ folder")
            return
    evaluator = AttackEvaluator(output_dir="results/comprehensive_attacks")
    report = evaluator.run_comprehensive_test(str(image_path))
    print("Summary:")
    print(f"  Baseline detections: {report['baseline_detections']}")
    print(f"  {report['summary']}")
    print()


def example_5_parametric_scan():
    print("Example 5: Parameter Scan for FGSM")
    print("-" * 40)
    image_path = find_image("test.jpg")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        print("\nMake sure test.jpg exists in data/ folder")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    wb = WhiteBoxAttacks()
    Path("results").mkdir(exist_ok=True)
    epsilons = [0.01, 0.05, 0.1, 0.2]
    print("Testing different epsilon values for FGSM:")
    for epsilon in epsilons:
        adversarial = wb.fgsm_attack(image_rgb, epsilon=epsilon)
        output_path = f"results/example5_fgsm_eps_{epsilon}.png"
        cv2.imwrite(output_path, cv2.cvtColor(adversarial, cv2.COLOR_RGB2BGR))
        print(f"  ✓ epsilon={epsilon:4.2f} > {output_path}")
    print()


def example_6_attack_sequence():
    print("Example 6: Sequential Attack Chain")
    print("-" * 40)
    image_path = find_image("test.jpg")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        print("\nMake sure test.jpg exists in data/ folder")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    wb = WhiteBoxAttacks()
    bb = BlackBoxAttacks()
    Path("results").mkdir(exist_ok=True)
    step1 = wb.fgsm_attack(image_rgb, epsilon=0.05)
    cv2.imwrite("results/example6_step1_fgsm.png", cv2.cvtColor(step1, cv2.COLOR_RGB2BGR))
    print("✓ Step 1: FGSM attack applied")
    step2 = bb.random_noise_attack(step1, noise_level=0.05)
    cv2.imwrite("results/example6_step2_noise.png", cv2.cvtColor(step2, cv2.COLOR_RGB2BGR))
    print("✓ Step 2: Noise added to FGSM result")
    step3 = bb.patch_attack(step2, patch_size=30)
    cv2.imwrite("results/example6_step3_patch.png", cv2.cvtColor(step3, cv2.COLOR_RGB2BGR))
    print("✓ Step 3: Adversarial patch applied")
    print()


def print_menu():
    print("\n" + "="*50)
    print("Adversarial Attacks Examples")
    print("="*50)
    print("\nAvailable examples:")
    print("  1. Single FGSM attack")
    print("  2. Multiple black-box attacks")
    print("  3. Compare attack types")
    print("  4. Comprehensive evaluation (all attacks)")
    print("  5. Parameter scan")
    print("  6. Sequential attack chain")
    print("  0. Run all examples")
    print("="*50 + "\n")


if __name__ == "__main__":
    import sys
    load_config()
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print_menu()
        choice = input("Select example (0-6): ").strip()
    
    print()
    
    examples = {
        "1": example_1_single_attack,
        "2": example_2_black_box_attacks,
        "3": example_3_compare_attacks,
        "4": example_4_comprehensive_test,
        "5": example_5_parametric_scan,
        "6": example_6_attack_sequence,
    }
    
    if choice == "0":
        for example_func in examples.values():
            example_func()
    elif choice in examples:
        examples[choice]()
    else:
        print("Invalid choice!")
        sys.exit(1)
    
    print("\nAll results saved to: results/")
