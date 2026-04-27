from attack_eval import AttackEvaluator
from model_functions import detect
from coords_extractor import extract_attack_coordinates
eval = AttackEvaluator()
test_path = "../data/test.jpg"

detect_result = detect (input_path=test_path, save_images=False)
detect_result_sp = extract_attack_coordinates(detect_result, strategy="random",points_per_bbox=10, target_class="cell phone")

eval.run_single_attack(test_path,'single_pixel_attack', "black_box", {'num_modifications' : len(detect_result_sp), "pixel_coordinates": detect_result_sp}, output_dir="../results/single_pixel_attack_results")

detect_result_pt = extract_attack_coordinates(detect_result, target_class="cell phone", return_patch_info=True, patch_size_mode="fixed", patch_size_value=50)
eval.run_single_attack("../data/test.jpg",'patch_attack', "black_box", {"patch_coordinates": (detect_result_pt[0]["x"], detect_result_pt[0]["y"]), "patch_size":detect_result_pt[0]["size"]}, output_dir="../results/patch_attack_results")

#eval.run_single_attack(test_path, 'single_pixel_attack', "black_box", , output_dir="../results/single_pixel_attack_results")