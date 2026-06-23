import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pytorch_lightning as pl

class SiamPruningCallback(pl.Callback):
    def __init__(self, pruning_amount=0.3, prune_epoch_frequency=10):
        super().__init__()
        self.pruning_amount = pruning_amount
        self.prune_epoch_frequency = prune_epoch_frequency

    def _apply_pruning(self, model, make_permanent=False):
        """
        Scan the student model's backbone and apply structured pruning to
        convolutional layers (specifically the pointwise/projection convolutions).
        """
        # We prune pointwise convolutions in the MobileNetV2 backbone.
        # These are Conv2d layers with kernel size (1, 1) and groups == 1.
        pruned_count = 0
        for name, module in model.student.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                # Only prune non-depthwise convolutions to avoid breaking channel relationships
                if module.groups == 1 and module.kernel_size[0] > 0:
                    try:
                        # Apply structured pruning along the filter dimension (dim=0)
                        # We prune a percentage of the output filters
                        prune.ln_structured(
                            module, 
                            name="weight", 
                            amount=self.pruning_amount, 
                            n=1, 
                            dim=0
                        )
                        pruned_count += 1
                        
                        if make_permanent:
                            # Fuses the mask with the weight, making the pruning permanent
                            prune.remove(module, "weight")
                    except Exception as e:
                        # Some layers might have too few output filters or be otherwise incompatible
                        # We log and skip them
                        pass
        return pruned_count

    def on_train_end(self, trainer, pl_module):
        print(f"\n[Pruning] Applying structured pruning (amount={self.pruning_amount}) at the end of training...")
        
        # Apply pruning and make it permanent
        pruned_count = self._apply_pruning(pl_module, make_permanent=True)
        print(f"[Pruning] Successfully applied and finalized pruning on {pruned_count} layers.")
        
        # Save the pruned model checkpoint
        pruned_ckpt_path = trainer.checkpoint_callback.best_model_path.replace(".ckpt", "_pruned.ckpt")
        if not pruned_ckpt_path or pruned_ckpt_path == "":
            # Fallback if no checkpoint callback is set
            pruned_ckpt_path = "weights/best_model_pruned.ckpt"
            
        trainer.save_checkpoint(pruned_ckpt_path)
        print(f"[Pruning] Saved pruned model checkpoint to: {pruned_ckpt_path}")
