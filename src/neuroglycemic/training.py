"""Compatibility imports for the single production neural trainer.

New code must import :mod:`src.neuroglycemic.neural_training` directly.  This
module remains temporarily so older notebooks fail neither silently nor onto a
different checkpoint schema.
"""

from .neural_training import (  # noqa: F401
    CHECKPOINT_SCHEMA,
    GlucoseTargetStandardizer,
    LossOutput,
    NeuralTrainingConfig,
    NeuralTrainingResult,
    inverse_transform_neuroglycemic_outputs,
    load_neural_checkpoint,
    load_neural_training_config,
    make_neuroglycemic_loss_step,
    save_neural_checkpoint,
    train_with_early_stopping,
)

__all__ = [
    "CHECKPOINT_SCHEMA",
    "GlucoseTargetStandardizer",
    "LossOutput",
    "NeuralTrainingConfig",
    "NeuralTrainingResult",
    "inverse_transform_neuroglycemic_outputs",
    "load_neural_checkpoint",
    "load_neural_training_config",
    "make_neuroglycemic_loss_step",
    "save_neural_checkpoint",
    "train_with_early_stopping",
]
