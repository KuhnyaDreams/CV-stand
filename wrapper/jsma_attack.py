import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import tf_keras as k3
import os
#model = k3.models.load_model("../core/yolo26n_saved_model")
model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

PHONE_CLASS = 737

IMAGENET_MEAN = np.array([0.485, 0.494, 0.456])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
IMAGENET_STD_T = torch.tensor(IMAGENET_STD).view(3, 1, 1)

NORM_MIN = (0/255 - IMAGENET_MEAN) / IMAGENET_STD
NORM_MAX = (255/255 - IMAGENET_MEAN) / IMAGENET_STD

def denormalize_imagenet(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.clone().float()
    else:
        tensor = torch.from_numpy(tensor).float()
    if tensor.dim() == 4:
        tensor = tensor[0]
    mean = IMAGENET_MEAN_T.to(tensor.device)
    std = IMAGENET_STD_T.to(tensor.device)
    denorm = tensor * std + mean
    denorm = torch.clamp(denorm * 255, 0, 255)
    return denorm.permute(1, 2, 0).byte().cpu().numpy()

def save_adversarial_image(img_tensor, output_path):
    img_np = denormalize_imagenet(img_tensor)
    Image.fromarray(img_np).save(output_path)
    print(f"Saved: {os.path.abspath(output_path)}")

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    ])
    return transform(img).unsqueeze(0)

def load_original_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return torch.from_numpy(img_array).float()

def jsma_avoid_class(model, image_path, avoid_class, max_iter=100, 
                     theta=0.1, gamma=0.05, device='cpu'):
    model = model.to(device)
    model.eval()

    original_full = load_original_image(image_path).to(device)
    h, w = original_full.shape[2], original_full.shape[3]
    print(f"Image size: {w}x{h}")

    original_224 = F.interpolate(original_full, size=(224, 224), 
                                 mode='bilinear', align_corners=False)

    with torch.no_grad():
        init_output = model(original_224)
        init_prob = F.softmax(init_output, dim=1)[0, avoid_class].item()
        init_top1 = init_output.argmax(dim=1).item()
    print(f"Initial prob: {init_prob:.4f}, top1: {init_top1}")

    adv_224 = original_224.clone()
    modified_mask = torch.zeros_like(adv_224)

    for iteration in range(max_iter):
        adv_224 = adv_224.clone().detach().requires_grad_(True)
        output = model(adv_224)
        current_prob = F.softmax(output, dim=1)[0, avoid_class]

        if output.argmax(dim=1).item() != avoid_class:
            print(f"Attack successful at iteration {iteration}")
            break

        current_prob.backward()
        grad = adv_224.grad.clone()

        saliency = torch.abs(grad) * (modified_mask == 0)
        if saliency.max() == 0:
            print("No available pixels")
            break

        flat_saliency = saliency.view(-1)
        num_mod = min(5, (flat_saliency > 0).sum().item())
        if num_mod == 0:
            break

        top_indices = torch.topk(flat_saliency, k=num_mod).indices

        for idx in top_indices:
            if modified_mask.view(-1)[idx] == 0:
                direction = -torch.sign(grad.view(-1)[idx])
                new_val = adv_224.data.view(-1)[idx] + gamma * direction
                channel = (idx // (224 * 224)) % 3
                if NORM_MIN[channel] <= new_val <= NORM_MAX[channel]:
                    adv_224.data.view(-1)[idx] = new_val
                    modified_mask.view(-1)[idx] = 1

        if modified_mask.sum() >= int(theta * adv_224.numel()):
            print("L0 limit reached")
            break

        if iteration % 10 == 0:
            print(f"Iter {iteration+1}: prob={current_prob:.4f}, modified={modified_mask.sum().item()}")

    delta_224 = adv_224.detach() - original_224
    delta_full = F.interpolate(delta_224, size=(h, w), mode='nearest')
    adv_full = original_full + delta_full

    for c in range(3):
        adv_full[:, c, :, :] = torch.clamp(adv_full[:, c, :, :], NORM_MIN[c], NORM_MAX[c])

    print(f"Attack finished. Final size: {w}x{h}")
    return adv_full.cpu()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    IMAGE_PATH = '../data/test.png'

    adversarial_img = jsma_avoid_class(
        model=model,
        image_path=IMAGE_PATH,
        avoid_class=PHONE_CLASS,
        max_iter=100,
        theta=0.1,
        gamma=0.05,
        device=device
    )

    save_adversarial_image(adversarial_img, '../data/adversarial_no_phone.png')

    from model_functions import detect
    detect(input_path='test.png')
    detect(input_path='adversarial_no_phone.png')