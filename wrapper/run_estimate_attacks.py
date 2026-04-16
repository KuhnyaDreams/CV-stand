from attack_eval import AttackEvaluator

aEval = AttackEvaluator()

image, report = aEval.run_pose_estimate_attack("../data/test.jpg", 'black_box', 'patch_attack')
print(report)