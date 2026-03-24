#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from attacks import AttackEvaluator, find_image_path, load_config
import json


def print_header():
    print("\n" + "="*70)
    print("🎯 Comprehensive Adversarial Attack Analysis".center(70))
    print("="*70)


def print_footer():
    print("="*70)
    print("✅ Attack analysis completed!".center(70))
    print("="*70 + "\n")


def print_report(report: dict):
    print("\n" + "-"*70)
    print("📊 SUMMARY REPORT")
    print("-"*70)
    
    summary = report.get('summary', {})
    baseline = report.get('baseline_detections', 0)
    
    print(f"\n📷 Input Image: {report.get('input_image')}")
    print(f"⏰ Timestamp: {report.get('timestamp')}")
    print(f"\n🔍 Baseline Detection: {baseline} object(s)")
    
    print(f"\n🤍 White-Box Attacks: {summary.get('white_box_success_rate', 'N/A')}")
    print(f"⬛ Black-Box Attacks: {summary.get('black_box_success_rate', 'N/A')}")
    
    print(f"\n📈 Total Attacks: {summary.get('total_attacks', '?')}")
    print(f"✅ Successful Attacks: {summary.get('successful_attacks', '?')}")
    
    print("arbuz")
    success_rate = (summary.get('successful_attacks', 0) / summary.get('total_attacks', 1)) * 100
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    print("\n" + "-"*70)
    print("📁 Detailed Results:")
    print("-"*70)
    
    # White-box results
    print("\n🤍 White-Box Attacks Results:")
    for attack_name, result in report.get('white_box_attacks', {}).items():
        if isinstance(result, dict) and 'detections' in result:
            status = "✓ SUCCESS" if result.get('success') else "✗ FAILED"
            print(f"  • {attack_name:12} → {result['detections']} detections {status}")
        elif isinstance(result, dict) and 'error' in result:
            print(f"  • {attack_name:12} → ⚠️  ERROR: {result['error']}")
    
    # Black-box results
    print("\n⬛ Black-Box Attacks Results:")
    for attack_name, result in report.get('black_box_attacks', {}).items():
        if isinstance(result, dict) and 'detections' in result:
            status = "✓ SUCCESS" if result.get('success') else "✗ FAILED"
            print(f"  • {attack_name:18} → {result['detections']} detections {status}")
        elif isinstance(result, dict) and 'error' in result:
            print(f"  • {attack_name:18} → ⚠️  ERROR: {result['error']}")
    
    print("\n" + "-"*70)


def main():
    load_config()
    parser = argparse.ArgumentParser(
        description="Run comprehensive adversarial attack analysis on an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_attack.py                    # Use test.jpg from data/
  python run_full_attack.py image.jpg          # Use specific image
  python run_full_attack.py /path/to/image.png # Use full path
  python run_full_attack.py --output attacks   # Custom output directory
        """
    )
    
    parser.add_argument(
        'image',
        nargs='?',
        help='Path to image file (default: test.jpg from data/)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='results/attack_results',
        help='Output directory for results (default: results/attack_results)'
    )
    
    parser.add_argument(
        '--skip-report',
        action='store_true',
        help='Skip printing detailed report'
    )
    
    parser.add_argument(
        '--save-report',
        type=str,
        help='Save report to JSON file'
    )
    
    args = parser.parse_args()
    if args.image:
        image_path = args.image
    else:
        image_path = find_image_path("test.jpg")
    
    if not Path(image_path).exists():
        print(f"\n❌ Error: Image not found: {image_path}")
        print("\nTrying to find images in data/ folders...")
        found = False
        for data_dir_path in [Path("data"), Path("../data"), Path("../../data")]:
            if data_dir_path.exists():
                images = list(data_dir_path.glob("*.jpg")) + list(data_dir_path.glob("*.png"))
                if images:
                    image_path = str(images[0])
                    print(f"✓ Found: {image_path}")
                    found = True
                    break
        if not found:
            print("\n❌ No images found!")
            print("Please provide an image path or place images in data/ folder")
            sys.exit(1)
    
    print_header()
    print(f"\n📷 Image: {image_path}")
    print(f"📁 Output: {args.output}\n")
    
    try:
        evaluator = AttackEvaluator(output_dir=args.output)
        report = evaluator.run_comprehensive_test(str(image_path))
        if not args.skip_report:
            print_report(report)
        if args.save_report:
            with open(args.save_report, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Report saved to: {args.save_report}")
        print_footer()
        return 0
    except Exception as e:
        print(f"\n❌ Error occurred:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
