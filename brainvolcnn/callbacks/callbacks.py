import os.path as op

from pytorch_lightning.callbacks import Callback


class RCLossMarginTune(Callback):
    """Callback to tune the within and between margin of the RCLossAnneal loss function."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.loss_fn.update_margins(trainer.current_epoch)
        self.log("hp/within_margin", pl_module.loss_fn.within_margin)
        self.log("hp/between_margin", pl_module.loss_fn.between_margin)

    # Log reconstruction loss and contrastive loss at the end of each batch
    def on_training_epoch_end(self, trainer, pl_module):
        self.log("train/recon_loss", pl_module.loss_fn.recon_loss)
        self.log("train/cont_loss", pl_module.loss_fn.contrast_loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log("val/recon_loss", pl_module.loss_fn.recon_loss)
        self.log("val/cont_loss", pl_module.loss_fn.contrast_loss)


class SaveLastModel(Callback):
    """Callback to save the model at the end of training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_end(self, trainer, pl_module):
        trainer.save_checkpoint(op.join(trainer.default_root_dir, "last.ckpt"))


class LogGradients(Callback):
    """Callback to log the gradients of the model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        for tag, value in pl_module.named_parameters():
            if value.grad is not None:
                pl_module.logger.experiment.add_histogram(
                    f"{tag}/grad",
                    value.grad.data.cpu(),
                    global_step=trainer.global_step,
                )


class LogParameters(Callback):
    """Callback to log the parameters of the model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_epoch_end(self, trainer, pl_module):
        for tag, value in pl_module.named_parameters():
            pl_module.logger.experiment.add_histogram(
                f"{tag}/weight", value.data.cpu(), global_step=trainer.global_step
            )


# class LogPredictionVariance(Callback):
#     """Callback to log the variance of the predicted target."""

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def on_validation_epoch_end(self, trainer, pl_module):

#         for batch in trainer.val_dataloaders:
#             x, _ = batch
#             y_hat = pl_module(x)
#             self.logger.experiment.add_histogram(
#                 "predicted_target_variance",
#                 y_hat.std(dim=0).flatten(),
#                 global_step=trainer.global_step,
#             )
