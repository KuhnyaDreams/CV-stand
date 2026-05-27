import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Optional, Union, Tuple
import logging
from base_attacks import AttackBase
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
IMAGENET_STD_T = torch.tensor(IMAGENET_STD).view(3, 1, 1)

NORM_MIN = (0/255 - IMAGENET_MEAN) / IMAGENET_STD
NORM_MAX = (255/255 - IMAGENET_MEAN) / IMAGENET_STD


class WhiteBoxAttacks(AttackBase):
    """White-box adversarial attack implementations."""
    
    # ==================== HELPER FUNCTIONS ====================
    
    @staticmethod
    def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Tuple[torch.Tensor, Tuple]:
        """Load and preprocess image for PyTorch models (ImageNet normalization)."""
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
        ])
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor, img.size
    
    @staticmethod
    def load_original_image(image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Load image in original resolution."""
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
        img_array = np.transpose(img_array, (2, 0, 1))
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()
        return img_tensor, (img.size[1], img.size[0])  # (height, width)
    
    @staticmethod
    def denormalize_imagenet(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert normalized tensor to [0, 255] uint8 image."""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.clone().float()
        else:
            tensor = torch.from_numpy(tensor).float()
        
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        mean = IMAGENET_MEAN_T
        std = IMAGENET_STD_T
        denorm = tensor * std + mean
        denorm = torch.clamp(denorm * 255, 0, 255)
        return denorm.permute(1, 2, 0).byte().cpu().numpy()
    
    @staticmethod
    def save_adversarial_image(img_tensor: torch.Tensor, output_path: str) -> None:
        """Save adversarial image."""
        img_np = WhiteBoxAttacks.denormalize_imagenet(img_tensor)
        Image.fromarray(img_np).save(output_path)
        logger.info(f"Сохранено: {os.path.abspath(output_path)}")
    
    # ==================== FGSM ATTACK ====================
    
    def fgsm_attack_tf(
        self,
        model,
        images: tf.Tensor,
        labels: tf.Tensor,
        epsilon: float = 0.1
    ) -> tf.Tensor:
        """
        FGSM attack using TensorFlow.
        
        Args:
            model: TensorFlow/Keras model
            images: Input images tensor
            labels: True labels tensor
            epsilon: Perturbation magnitude
        
        Returns:
            adversarial_images: Perturbed images
        """
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels)
        
        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = model(images)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        
        gradients = tape.gradient(loss, images)
        signed_gradients = tf.sign(gradients)
        adversarial_images = images + epsilon * signed_gradients
        adversarial_images = tf.clip_by_value(adversarial_images, 0, 1)
        
        return adversarial_images
    
    def fgsm_avoid_class(
        self,
        model,
        image_path: str,
        avoid_class: int,
        epsilon: float = 1.0,
        max_iter: int = 15,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        FGSM attack to avoid a specific class (PyTorch).
        
        Args:
            model: PyTorch model
            image_path: Path to input image
            avoid_class: Class index to avoid
            epsilon: Perturbation magnitude per step
            max_iter: Number of iterations
            device: 'cpu' or 'cuda'
        
        Returns:
            Adversarial image tensor (full resolution, normalized)
        """
        model = model.to(device).eval()
        
        # Load original in full resolution
        original_full = WhiteBoxAttacks.load_original_image(image_path)[0].to(device)
        h, w = original_full.shape[2], original_full.shape[3]
        logger.info(f"📷 Размер изображения: {w}x{h}")
        
        # Resize for model
        original_224 = F.interpolate(original_full, size=(224, 224), mode='bilinear', align_corners=False)
        adv_224 = torch.autograd.Variable(original_224.clone(), requires_grad=True)
        
        with torch.no_grad():
            init_output = model(original_224)
            init_prob = F.softmax(init_output, dim=1)[0, avoid_class].item()
        logger.info(f"📉 Исходная вероятность класса {avoid_class}: {init_prob:.4f}")
        
        for i in range(max_iter):
            if adv_224.grad is not None:
                adv_224.grad.zero_()
            
            predictions = model(adv_224)
            loss = F.cross_entropy(predictions, torch.tensor([avoid_class]).to(device))
            loss.backward()
            
            adv_224.data = adv_224.data + epsilon * torch.sign(adv_224.grad.data)
            adv_224.data = torch.clamp(adv_224.data, -125.0/255.0, 155.0/255.0)
            
            if i % 3 == 0:
                with torch.no_grad():
                    current_prob = F.softmax(model(adv_224), dim=1)[0, avoid_class].item()
                logger.info(f"   Итерация {i+1}: вероятность = {current_prob:.4f}")
        
        # Scale perturbation back to original resolution
        delta_224 = adv_224 - original_224
        delta_full = F.interpolate(delta_224, size=(h, w), mode='bilinear', align_corners=False)
        adv_full = original_full + delta_full.detach()
        adv_full = torch.clamp(adv_full, -125.0/255.0, 155.0/255.0)
        
        logger.info(f"✅ FGSM завершена. Размер: {w}x{h}")
        return adv_full
    
    # ==================== PGD ATTACK ====================
    
    def pgd_avoid_class(
        self,
        model,
        image_path: str,
        avoid_class: int,
        epsilon: float = 1.0,
        max_iter: int = 15,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        PGD attack to avoid a specific class (PyTorch).
        
        Args:
            model: PyTorch model
            image_path: Path to input image
            avoid_class: Class index to avoid
            epsilon: Perturbation magnitude per step
            max_iter: Number of iterations
            device: 'cpu' or 'cuda'
        
        Returns:
            Adversarial image tensor (full resolution, normalized)
        """
        model = model.to(device).eval()
        
        original_full = WhiteBoxAttacks.load_original_image(image_path)[0].to(device)
        h, w = original_full.shape[2], original_full.shape[3]
        logger.info(f"📷 Размер изображения: {w}x{h}")
        
        original_224 = F.interpolate(original_full, size=(224, 224), mode='bilinear', align_corners=False)
        adv_224 = original_224.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([adv_224], lr=epsilon)
        
        with torch.no_grad():
            init_output = model(original_224)
            init_prob = F.softmax(init_output, dim=1)[0, avoid_class].item()
        logger.info(f"📉 Исходная вероятность класса {avoid_class}: {init_prob:.4f}")
        
        for i in range(max_iter):
            optimizer.zero_grad()
            predictions = model(adv_224)
            loss = F.cross_entropy(predictions, torch.tensor([avoid_class]).to(device))
            loss.backward()
            
            adv_224.data = adv_224.data + epsilon * torch.sign(adv_224.grad.data)
            adv_224.data = torch.clamp(adv_224.data, -125.0/255.0, 155.0/255.0)
            
            if i % 3 == 0:
                with torch.no_grad():
                    current_prob = F.softmax(model(adv_224), dim=1)[0, avoid_class].item()
                logger.info(f"   Итерация {i+1}: вероятность = {current_prob:.4f}")
        
        delta_224 = adv_224 - original_224
        delta_full = F.interpolate(delta_224, size=(h, w), mode='bilinear', align_corners=False)
        adv_full = original_full + delta_full.detach()
        adv_full = torch.clamp(adv_full, -125.0/255.0, 155.0/255.0)
        
        logger.info(f"✅ PGD завершена. Размер: {w}x{h}")
        return adv_full
    
    # ==================== JSMA ATTACK ====================
    
    def jsma_avoid_class(
        self,
        model,
        image_path: str,
        avoid_class: int,
        max_iter: int = 100,
        theta: float = 0.1,
        gamma: float = 0.05,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        JSMA attack to avoid a specific class (PyTorch).
        
        Args:
            model: PyTorch model
            image_path: Path to input image
            avoid_class: Class index to avoid
            max_iter: Maximum iterations
            theta: Perturbation magnitude
            gamma: Saliency threshold
            device: 'cpu' or 'cuda'
        
        Returns:
            Adversarial image tensor (full resolution, normalized)
        """
        model = model.to(device).eval()
        
        original_full = WhiteBoxAttacks.load_original_image(image_path)[0].to(device)
        h, w = original_full.shape[2], original_full.shape[3]
        logger.info(f"📷 Размер изображения: {w}x{h}")
        
        original_224 = F.interpolate(original_full, size=(224, 224), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            init_output = model(original_224)
            init_prob = F.softmax(init_output, dim=1)[0, avoid_class].item()
        logger.info(f"📉 Исходная вероятность класса {avoid_class}: {init_prob:.4f}")
        
        adv_224 = original_224.clone()
        modified_mask = torch.zeros_like(adv_224)
        
        for iteration in range(max_iter):
            adv_224_var = adv_224.clone().detach().requires_grad_(True)
            output = model(adv_224_var)
            current_prob = F.softmax(output, dim=1)[0, avoid_class]
            
            if output.argmax(dim=1).item() != avoid_class:
                logger.info(f"🎯 Атака успешна на итерации {iteration}")
                break
            
            current_prob.backward()
            grad = adv_224_var.grad.clone()
            
            saliency = torch.abs(grad) * (modified_mask == 0)
            if saliency.max() == 0:
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
                logger.info("⚠️ Достигнут лимит L0")
                break
            
            if iteration % 10 == 0:
                logger.info(f"Итер {iteration+1}: prob={current_prob:.4f}, modified={modified_mask.sum().item()}")
        
        delta_224 = adv_224 - original_224
        delta_full = F.interpolate(delta_224, size=(h, w), mode='nearest')
        adv_full = original_full + delta_full
        
        for c in range(3):
            adv_full[:, c, :, :] = torch.clamp(adv_full[:, c, :, :], NORM_MIN[c], NORM_MAX[c])
        
        logger.info(f"✅ JSMA завершена. Размер: {w}x{h}")
        return adv_full.detach()
    
    # ==================== DEEPFOOL ATTACK ====================
    
    def deepfool_avoid_class(
        self,
        model,
        image_path: str,
        avoid_class: int,
        num_classes: int = 10,
        overshoot: float = 0.02,
        max_iter: int = 50,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        DeepFool attack to avoid a specific class (PyTorch).
        
        Args:
            model: PyTorch model
            image_path: Path to input image
            avoid_class: Class index to avoid
            num_classes: Number of neighboring classes to consider
            overshoot: Overshoot coefficient
            max_iter: Maximum iterations
            device: 'cpu' or 'cuda'
        
        Returns:
            Adversarial image tensor (full resolution, normalized)
        """
        model = model.to(device).eval()
        
        original_full = WhiteBoxAttacks.load_original_image(image_path)[0].to(device)
        h, w = original_full.shape[2], original_full.shape[3]
        logger.info(f"📷 Размер изображения: {w}x{h}")
        
        image_224 = F.interpolate(original_full, size=(224, 224), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            init_output = model(image_224)
            init_prob = F.softmax(init_output, dim=1)[0, avoid_class].item()
        logger.info(f"📉 Исходная вероятность класса {avoid_class}: {init_prob:.4f}")
        
        input_shape = image_224.shape
        r_tot = np.zeros(input_shape)
        pert_image = image_224.clone()
        
        loop_i = 0
        while loop_i < max_iter:
            pert_image_var = pert_image.clone().detach().requires_grad_(True)
            output = model(pert_image_var)
            current_class = torch.argmax(output).item()
            
            if current_class != avoid_class:
                logger.info(f"🎯 DeepFool успешна на итерации {loop_i}")
                break
            
            probs = F.softmax(output, dim=1)
            top_k = min(num_classes + 1, output.shape[1])
            top_probs, top_indices = torch.topk(probs[0], k=top_k)
            
            output[0, avoid_class].backward(retain_graph=True)
            grad_orig = pert_image_var.grad.detach().cpu().numpy().copy()
            pert_image_var.grad.zero_()
            
            pert = np.inf
            w_best = None
            
            for idx in top_indices:
                k = idx.item()
                if k == avoid_class:
                    continue
                
                output[0, k].backward(retain_graph=True)
                grad_k = pert_image_var.grad.detach().cpu().numpy().copy()
                pert_image_var.grad.zero_()
                
                w_k = grad_k - grad_orig
                f_k = (output[0, k] - output[0, avoid_class]).detach().cpu().numpy()
                pert_k = np.abs(f_k) / (np.linalg.norm(w_k.flatten()) + 1e-8)
                
                if pert_k < pert:
                    pert = pert_k
                    w_best = w_k
            
            if w_best is None:
                break
            
            norm_w = np.linalg.norm(w_best) + 1e-8
            r_i = (pert + 1e-4) * w_best / norm_w
            r_tot = r_tot + r_i
            
            pert_image = image_224 + torch.from_numpy((1 + overshoot) * r_tot).float().to(device)
            pert_image = torch.clamp(pert_image, -3, 3)
            
            loop_i += 1
        
        r_tot = (1 + overshoot) * r_tot
        adv_224 = image_224 + torch.from_numpy(r_tot).float().to(device)
        adv_224 = torch.clamp(adv_224, -3, 3)
        
        delta_224 = adv_224 - image_224
        delta_full = F.interpolate(delta_224, size=(h, w), mode='bilinear', align_corners=False)
        adv_full = original_full + delta_full
        adv_full = torch.clamp(adv_full, -3, 3)
        
        logger.info(f"✅ DeepFool завершена. Размер: {w}x{h}")
        return adv_full
    
    # ==================== LEGACY METHODS (для совместимости) ====================
    
    def pgd_attack(
        self,
        image: np.ndarray,
        epsilon: Optional[float] = None,
        num_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Projected Gradient Descent attack (legacy numpy version).
        """
        self.validate_image(image)
        
        if epsilon is None:
            epsilon = self.get_config_param('white_box_attacks', 'pgd', 'epsilon', 0.03)
        if num_steps is None:
            num_steps = self.get_config_param('white_box_attacks', 'pgd', 'num_steps', 7)
        
        self.log_attack('pgd_attack', epsilon=epsilon, num_steps=num_steps)
        
        image_normalized = self.normalize_to_unit(image)
        adversarial = image_normalized.copy()
        
        for step in range(num_steps):
            noise = np.random.randn(*image.shape) * (epsilon / num_steps)
            adversarial_normalized = adversarial + noise
            perturbation = adversarial_normalized - image_normalized
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            adversarial = image_normalized + perturbation
            adversarial = self.clip_image(adversarial)
        
        return self.denormalize_to_uint8(adversarial)
    
    def deepfool_attack(
        self,
        image: np.ndarray,
        num_classes: Optional[int] = None
    ) -> np.ndarray:
        """
        DeepFool attack - minimal adversarial perturbation (legacy numpy version).
        """
        self.validate_image(image)
        
        if num_classes is None:
            num_classes = self.get_config_param('white_box_attacks', 'deepfool', 'num_classes', 80)
        
        self.log_attack('deepfool_attack', num_classes=num_classes)
        
        image_normalized = self.normalize_to_unit(image)
        perturbation = np.random.randn(*image.shape) * 0.02
        adversarial = self.clip_image(image_normalized + perturbation)
        return self.denormalize_to_uint8(adversarial)
    
    def jsma_attack(
        self,
        image: np.ndarray,
        theta: Optional[float] = None,
        gamma: Optional[float] = None
    ) -> np.ndarray:
        """
        Jacobian-based Saliency Map Attack (legacy numpy version).
        """
        self.validate_image(image)
        
        if theta is None:
            theta = self.get_config_param('white_box_attacks', 'jsma', 'theta', 1.0)
        if gamma is None:
            gamma = self.get_config_param('white_box_attacks', 'jsma', 'gamma', 0.1)
        
        self.log_attack('jsma_attack', theta=theta, gamma=gamma)
        
        image_normalized = self.normalize_to_unit(image)
        h, w = image.shape[:2]
        saliency = np.random.rand(h, w) * gamma
        
        num_pixels = max(1, int(h * w * 0.01))
        for _ in range(num_pixels):
            y, x = np.unravel_index(np.argmax(saliency), saliency.shape)
            perturbed = self.clip_image(image_normalized[y, x] + theta / 255.0)
            image_normalized[y, x] = perturbed[0] if isinstance(perturbed, np.ndarray) else perturbed
        
        return self.denormalize_to_uint8(image_normalized)