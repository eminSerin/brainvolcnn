from pytorch_lightning.callbacks import Callback


class RCLossMarginTune(Callback):
    """Callback to tune the within and between margin of the RCLossAnneal loss function."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.loss_fn.update_margins(trainer.current_epoch)
        self.log("hp/within_margin", pl_module.loss_fn.within_margin)
        self.log("hp/between_margin", pl_module.loss_fn.between_margin)
