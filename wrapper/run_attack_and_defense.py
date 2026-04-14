from defence import DefensePipeline, Defenses
from attack_eval import AttackEvaluator
from detection_functions import detect_image

aEval = AttackEvaluator()
defenses = Defenses()
dPipeline = DefensePipeline( defenses.combined)
def run_attack_and_defense(image_path: str):
    image, report = aEval.run_single_attack_image(image_path, 'black_box', 'gaussian_blur_attack')
    print(report)
    defended_image = dPipeline.apply(image)
    report = detect_image(defended_image)
    print("====================")
    print(report)

run_attack_and_defense("data/test.jpg")