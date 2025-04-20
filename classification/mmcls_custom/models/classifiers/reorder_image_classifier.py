import torch
import torch.nn as nn

from mmcls.models.builder import CLASSIFIERS, MODELS, build_classifier
from mmcls.models.classifiers.base import BaseClassifier

# Assume you have defined and registered your Reorder module elsewhere, e.g., in mmcls.models.modules
# from ..modules import Reorder # Import your Reorder module

# Register the new top-level classifier
@CLASSIFIERS.register_module()
class ReorderImageClassifier(BaseClassifier):

    # Accept configs for the Reorder module and the inner ImageClassifier,
    # plus the path to the pre-trained checkpoint for the inner classifier.
    def __init__(self,
                 reorder_module_cfg,
                 image_classifier_module_cfg,
                 pretrained_checkpoint_path=None,
                 reg_weight=0.0,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)

        if reorder_module_cfg.get('type') is None:
             raise ValueError("Reorder module config must specify 'type'")
        else:
             self.reorder_module = MODELS.build(reorder_module_cfg)

        self.image_classifier_module = build_classifier(image_classifier_module_cfg)

        self.reg_weight = reg_weight 

        if pretrained_checkpoint_path is not None:
            self.load_pretrained_image_classifier(pretrained_checkpoint_path)

        self._freeze_image_classifier()

    def load_pretrained_image_classifier(self, checkpoint_path):
        """Load pre-trained weights for the image_classifier_module."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # Check if the checkpoint state_dict is nested
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Optional: Handle potential prefix mismatches if necessary
            # Example: if state_dict keys are 'module.backbone.param', but inner module expects 'backbone.param'
            # new_state_dict = {}
            # for k, v in state_dict.items():
            #     if k.startswith('module.'): # Example prefix
            #          new_state_dict[k[7:]] = v
            #     else:
            #          new_state_dict[k] = v
            # state_dict = new_state_dict

            # Load the state dict into the inner module
            # Set strict=True if you expect an exact match, False otherwise (use with caution)
            msg = self.image_classifier_module.load_state_dict(state_dict, strict=True)
            print(f"Loaded pretrained weights for image_classifier_module from {checkpoint_path}")
            # print(msg) # Uncomment to see missing/unexpected keys

        except Exception as e:
            print(f"Error loading pretrained checkpoint {checkpoint_path}: {e}")
            # Depending on requirements, you might want to raise the exception
            # raise e


    def _freeze_image_classifier(self):
        """Freeze all parameters of the inner image_classifier_module."""
        print("Freezing parameters of the inner image_classifier_module.")
        for param in self.image_classifier_module.parameters():
            param.requires_grad = False
        # Optional: Also freeze buffers if they exist and shouldn't be updated (e.g., running stats in BN)
        # self.image_classifier_module.eval() # Setting eval mode often helps prevent BN updates


    # Standard mmcls forward method to dispatch to train/test
    def forward(self, img, return_loss=True, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
            return_loss (bool): Whether to return loss. Defaults to True.
            kwargs (dict): Other keyword arguments for the specific task.
                - gt_label (Tensor): Ground truth labels for training.
        """
        if return_loss:
            # In training mode, forward_train is called
            return self.forward_train(img, **kwargs)
        else:
            # In evaluation/inference mode, simple_test or aug_test is called
            return self.simple_test(img, **kwargs)
            # Or if you implement aug_test:
            # if 'aug_data' in kwargs:
            #     return self.aug_test(kwargs['aug_data'])
            # else:
            #     return self.simple_test(img, **kwargs)


    def forward_train(self, img, gt_label, **kwargs):
        """
        Forward function for training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
            gt_label (Tensor): Ground truth labels.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Ensure the inner classifier is in eval mode even during training
        # to prevent BN updates if they weren't fully frozen.
        self.image_classifier_module.eval()

        reordered_img, P = self.reorder_module(img)

        losses = self.image_classifier_module.forward_train(reordered_img, gt_label, **kwargs)

        if self.reg_weight > 0:
            N, N_patches, _ = P.shape
            I = torch.eye(N_patches, device=P.device).unsqueeze(0).expand(N, -1, -1)
            regularization_loss = torch.mean((P - I)**2)

            losses['regularization'] = (regularization_loss - 0.002)**2 * self.reg_weight
            losses['classification'] = losses['loss']
            # 更新总损失 'loss' (mmcls runner 默认优化 'loss' 键)
            if isinstance(losses['loss'], torch.Tensor):
                losses['loss'] = losses['loss'] + losses['regularization']

        return losses

    def simple_test(self, img, **kwargs):
        """
        Forward function for inference.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).

        Returns:
            list[np.ndarray]: Predicted results.
        """
        self.eval()

        with torch.no_grad():
            reordered_img, P = self.reorder_module(img)
            results = self.image_classifier_module.simple_test(reordered_img, **kwargs)

        return results

    # You might also need to implement aug_test if you use test-time augmentation
    # def aug_test(self, aug_data):
    #     # Implement logic for test-time augmentation if needed
    #     pass


    def extract_feat(self, img):
        """Directly extract features from the reordered images.

        Args:
            img (Tensor): The input images.

        Returns:
            tuple[Tensor]: The feature tensor (or tuple of tensors).
        """
        reordered_img, P = self.reorder_module(img)
        self.image_classifier_module.eval()
        with torch.no_grad():
            inner_features = self.image_classifier_module.extract_feat(reordered_img)

        return inner_features