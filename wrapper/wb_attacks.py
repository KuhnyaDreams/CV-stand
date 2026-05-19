import numpy as np
from typing import Optional,  Union, Tuple
import logging
from base_attacks import AttackBase
import tensorflow as tf
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


class WhiteBoxAttacks(AttackBase):
    """White-box adversarial attack implementations."""
    
    


    def fgsm_attack(model, images, labels, epsilon=0.1):
        """
        Generate FGSM adversarial examples
        
        Args:
            model: TensorFlow/Keras model
            images: Input images tensor
            labels: True labels tensor
            epsilon: Perturbation magnitude
        
        Returns:
            adversarial_images: Perturbed images
        """
        
        # Convert to tensor if numpy array
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels)
        
        # Record gradients
        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = model(images)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        
        # Calculate gradients
        gradients = tape.gradient(loss, images)
        
        # Generate adversarial examples
        signed_gradients = tf.sign(gradients)
        adversarial_images = images + epsilon * signed_gradients
        
        # Clip to valid pixel range [0, 1]
        adversarial_images = tf.clip_by_value(adversarial_images, 0, 1)
        
        return adversarial_images


    
    def pgd_attack(
        self,
        image: np.ndarray,
        epsilon: Optional[float] = None,
        num_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Projected Gradient Descent attack.
        
        Args:
            image: Input image array
            epsilon: Maximum perturbation
            num_steps: Number of iteration steps
            
        Returns:
            Adversarial image
        """
        self.validate_image(image)
        
        if epsilon is None:
            epsilon = self.get_config_param(
                'white_box_attacks',
                'pgd',
                'epsilon',
                0.03
            )
        
        if num_steps is None:
            num_steps = self.get_config_param(
                'white_box_attacks',
                'pgd',
                'num_steps',
                7
            )
        
        self.log_attack(
            'pgd_attack',
            epsilon=epsilon,
            num_steps=num_steps
        )
        
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
        DeepFool attack - minimal adversarial perturbation.
        
        Args:
            image: Input image array
            num_classes: Number of classes
            
        Returns:
            Adversarial image
        """
        self.validate_image(image)
        
        if num_classes is None:
            num_classes = self.get_config_param(
                'white_box_attacks',
                'deepfool',
                'num_classes',
                80
            )
        
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
        Jacobian-based Saliency Map Attack.
        
        Args:
            image: Input image array
            theta: Perturbation per pixel
            gamma: Saliency threshold
            
        Returns:
            Adversarial image
        """
        self.validate_image(image)
        
        if theta is None:
            theta = self.get_config_param(
                'white_box_attacks',
                'jsma',
                'theta',
                1.0
            )
        
        if gamma is None:
            gamma = self.get_config_param(
                'white_box_attacks',
                'jsma',
                'gamma',
                0.1
            )
        
        self.log_attack('jsma_attack', theta=theta, gamma=gamma)
        
        image_normalized = self.normalize_to_unit(image)
        h, w = image.shape[:2]
        saliency = np.random.rand(h, w) * gamma
        
        # Perturb pixels with highest saliency
        num_pixels = max(1, int(h * w * 0.01))
        for _ in range(num_pixels):
            y, x = np.unravel_index(np.argmax(saliency), saliency.shape)
            image_normalized[y, x] = self.clip_image(
                image_normalized[y, x] + theta / 255.0
            )[0] if isinstance(self.clip_image(image_normalized[y, x] + theta / 255.0), np.ndarray) else self.clip_image(
                image_normalized[y, x] + theta / 255.0
            )
        
        return self.denormalize_to_uint8(image_normalized)