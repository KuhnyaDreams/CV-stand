from attack_eval import AttackEvaluator
eval = AttackEvaluator()

#eval.run_single_attack("../data/test.jpg",'single_pixel_attack', "black_box", {'num_modifications' : 100}, output_dir="../results/single_pixel_attack_results")
#eval.run_single_attack("../data/test.jpg",'patch_attack', "black_box", {"patch_coordinates": (290, 220), "patch_size":50}, output_dir="../results/patch_attack_results")

eval.run_single_attack("../data/test.jpg",'single_pixel_attack', "black_box", {'num_modifications' : 100}, output_dir="../results/single_pixel_attack_results")