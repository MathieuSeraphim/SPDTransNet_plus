from typing import Optional
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BaseFinetuning
from torch.optim import Optimizer
from _4_models.ModelWrapper import get_model_from_config_file
from _4_models._4_1_sequence_based_models.SPDFromEEGSuccessiveChannelsLearnedAugmentationTransformerModel import \
    SPDFromEEGSuccessiveChannelsLearnedAugmentationTransformerModel
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.SPDFromEEGDataMultichannelSuccessionLearnedAugmentationReformattingBlock import \
    SPDFromEEGDataMultichannelSuccessionLearnedAugmentationReformattingBlock
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.SPDFromEEGDataMultichannelSuccessionLearnedAugmentationWithRunningAverageReformattingBlock import \
    SPDFromEEGDataMultichannelSuccessionLearnedAugmentationWithRunningAverageReformattingBlock
from _4_models._4_1_sequence_based_models.data_formatting_block.SPD_from_EEG_data_reformatting.layers.MatrixAugmentationThroughComputedFeaturesLayer import \
    MatrixAugmentationThroughComputedFeaturesLayer
from _4_models._4_2_signal_feature_extractor_models.extractor_model.deepsleepnet_feature_extractor.MultichannelDeepSleepNetFeatureExtractorModel import \
    MultichannelDeepSleepNetFeatureExtractorModel
from _4_models._4_2_signal_feature_extractor_models.extractor_model.iitnet_feature_extractor.MultichannelIITNetFeatureExtractorModel import \
    MultichannelIITNetFeatureExtractorModel
from _4_models._4_2_signal_feature_extractor_models.extractor_model.iitnet_feature_extractor.MultichannelIITNetFeatureExtractorModel_v2 import \
    MultichannelIITNetFeatureExtractorModel_v2
from _4_models._4_2_signal_feature_extractor_models.extractor_model.zhu_et_al_feature_extractor.MultichannelWindowFeatureLearningModel import \
    MultichannelWindowFeatureLearningModel


class LearnedAugmentationFinetuning(BaseFinetuning):

    COMPATIBLE_MODELS = (SPDFromEEGSuccessiveChannelsLearnedAugmentationTransformerModel,)
    COMPATIBLE_MODEL_BLOCKS = (SPDFromEEGDataMultichannelSuccessionLearnedAugmentationReformattingBlock,
                               SPDFromEEGDataMultichannelSuccessionLearnedAugmentationWithRunningAverageReformattingBlock)
    COMPATIBLE_BLOCK_LAYERS = (MatrixAugmentationThroughComputedFeaturesLayer,)
    COMPATIBLE_EXTRACTOR_MODELS = (MultichannelDeepSleepNetFeatureExtractorModel,
                                   MultichannelIITNetFeatureExtractorModel,
                                   MultichannelIITNetFeatureExtractorModel_v2,
                                   MultichannelWindowFeatureLearningModel)

    SUCCESSIVE_ATTRIBUTES_DICT = {
        "SPDFromEEGSuccessiveChannelsLearnedAugmentationTransformerModel": {
            "attribute": "data_formatting_block",
            "corresponding_classes": {
                "SPDFromEEGDataMultichannelSuccessionLearnedAugmentationReformattingBlock": {
                    "attribute": "augmentation_layer",
                    "corresponding_classes": {
                        "MatrixAugmentationThroughComputedFeaturesLayer": {
                            "attribute": "feature_extractor_model"
                        }
                    }
                }
            }
        },
    }

    def __init__(self, unfreeze_at_epoch: int = 1, override_extractor_lr_with_value: Optional[float] = None,
                 divide_extractor_lr_by_value_if_not_overridden: float = 10.0):
        super(LearnedAugmentationFinetuning, self).__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.override_lr_value = override_extractor_lr_with_value
        self.lr_denom_if_not_overridden = divide_extractor_lr_by_value_if_not_overridden

    def get_extractor_model(self, model: LightningModule):
        model_name = model.__class__.__name__
        assert type(model) in self.COMPATIBLE_MODELS and model_name in self.SUCCESSIVE_ATTRIBUTES_DICT.keys()
        model_attribute_dict = self.SUCCESSIVE_ATTRIBUTES_DICT[model_name]

        model_block_name_as_attribute = model_attribute_dict["attribute"]
        model_block = getattr(model, model_block_name_as_attribute)
        model_block_name = model_block.__class__.__name__
        assert type(model_block) in self.COMPATIBLE_MODEL_BLOCKS\
               and model_block_name in model_attribute_dict["corresponding_classes"].keys()
        model_block_attribute_dict = model_attribute_dict["corresponding_classes"][model_block_name]

        block_layer_name_as_attribute = model_block_attribute_dict["attribute"]
        block_layer = getattr(model_block, block_layer_name_as_attribute)
        block_layer_name = block_layer.__class__.__name__
        assert type(block_layer) in self.COMPATIBLE_BLOCK_LAYERS \
               and block_layer_name in model_block_attribute_dict["corresponding_classes"].keys()
        block_layer_attribute_dict = model_block_attribute_dict["corresponding_classes"][block_layer_name]

        extractor_model_name_as_attribute = block_layer_attribute_dict["attribute"]
        extractor_model = getattr(block_layer, extractor_model_name_as_attribute)
        assert type(extractor_model) in self.COMPATIBLE_EXTRACTOR_MODELS

        return extractor_model

    def freeze_before_training(self, pl_module: LightningModule):
        extractor_model = self.get_extractor_model(pl_module)
        self.freeze(extractor_model)

    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        if epoch == self._unfreeze_at_epoch:
            extractor_model = self.get_extractor_model(pl_module)
            self.unfreeze_and_add_param_group(
                modules=extractor_model, optimizer=optimizer, lr=self.override_lr_value,
                initial_denom_lr=self.lr_denom_if_not_overridden)


if __name__ == "__main__":
    callback = LearnedAugmentationFinetuning()
    model_config = "EUSIPCO_signals_spd_preserving_network_learned_augmentation_length_21_config.yaml"
    model = get_model_from_config_file(model_config)
    model.setup("test")
    extractor = callback.get_extractor_model(model)
    assert type(extractor) in callback.COMPATIBLE_EXTRACTOR_MODELS




