from torch import nn

import shard_modules as fm
import utils


class ModelConfig(utils.SelfDescripting):
    """
    ModelConfig defines an interface followed by configuration classes of all models. 
    """

    def make_model(self) -> nn.Module:
        """
        Constructs a model according to this config
        """
        raise NotImplementedError()

    def no_prebuilt(self) -> 'ModelConfig':
        """
        Turns of the loading of pretrained weights
        """
        raise NotImplementedError()


class ShardModelConfig(ModelConfig):
    """
    ShardModelConfig extends the ModelConfig interface.
    ShardModelConfig is followed by configuration classes of all SHARD models. 
    """

    def make_model(self) -> fm.Module:
        """
        Constructs a model according to this config
        """
        raise NotImplementedError()

    def no_prebuilt(self) -> 'ShardModelConfig':
        """
        Turns of the loading of pretrained weights
        """
        raise NotImplementedError()

    def create_base_config(self, level) -> ModelConfig:
        """
        Creates a regular model config for the plain counterpart of this model.
        """
        raise NotImplementedError()

    def max_level(self) -> int:
        """
        Queries the maximum level the SHARD model will have.
        """
        raise NotImplementedError()
