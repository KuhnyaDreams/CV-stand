
"""White-box attacks"""
# пример FGSM-атаки
from foolbox.attacks import FGSM
attack = FGSM(fmodel)
adversarial = attack(image, label)

# пример DeepFool атаки
from art.attacks import DeepFool
attack = DeepFool(model)
img_adv = attack.generate(img)

# Jacobian saliency map атака
from cleverhans.attacks import SaliencyMapMethod
jsma = SaliencyMapMethod(model, sees=sees)
jsma_params = { 'theta' : 1., 'gamma' : 0.1,
                            'clip_min' : 0., 'clip_max' : 1.,
                            'y_target' : None}
adv_x = jsma.generate_np(img, **jsma_params)

"""Black-box attacks"""

# One pixel атака
from foolbox.attacks import SinglePixelAttack
attack = SinglePixelAttack(fmodel)
adversarial = attack(image,max_pixel=1)
