import warnings
from typing import List, Dict, Any, Union
from _4_models.BaseModel import BaseModel
from _4_models._4_2_signal_feature_extractor_models.extractor_model.BaseExtractorModel import BaseExtractorModel
from _4_models._4_2_signal_feature_extractor_models.features_to_classification_block.BaseFeaturesToClassificationBlock import \
    BaseFeaturesToClassificationBlock
from _4_models.utils import check_block_import


class FeatureExtractorPretrainingBaseModel(BaseModel):

    COMPATIBLE_EXTRACTOR_MODELS = ()
    COMPATIBLE_FEATURES_TO_CLASSIFICATION_BLOCKS = ()

    def __init__(self, loss_function_config_dict: Dict[str, Any], class_labels_list: List[str],
                 extractor_model: BaseExtractorModel,
                 features_to_classification_block: BaseFeaturesToClassificationBlock, learning_rate: float,
                 optimisation_config_dict: Union[Dict[str, Any], None] = None):
        super(FeatureExtractorPretrainingBaseModel, self).__init__(loss_function_config_dict, class_labels_list,
                                                                   learning_rate, optimisation_config_dict)
        self.__setup_done_flag = False

        self.extractor_model = check_block_import(extractor_model)
        self.features_to_classification_block = check_block_import(features_to_classification_block)

        assert isinstance(extractor_model, self.COMPATIBLE_EXTRACTOR_MODELS)
        assert isinstance(features_to_classification_block, self.COMPATIBLE_FEATURES_TO_CLASSIFICATION_BLOCKS)

        self.extractor_setup_kwargs = None
        self.features_to_classification_setup_kwargs = None

    def setup(self, stage: str, no_logging: bool = False):
        self.obtain_example_input_array()
        if not self.__setup_done_flag:
            self.extractor_model.setup(**self.extractor_setup_kwargs)
            self.features_to_classification_block.setup(**self.features_to_classification_setup_kwargs)
            self.__setup_done_flag = True

    def preprocess_input(self, input, set_name):
        raise NotImplementedError

    def forward(self, **inputs):
        assert self.__setup_done_flag
        features = self.extractor_model(**inputs)
        classification_logits = self.features_to_classification_block(features)
        return classification_logits

    def obtain_example_input_array(self):
        if not type(self) == FeatureExtractorPretrainingBaseModel:
            raise NotImplementedError
        warnings.warn("This is a base instance of a pretraining model for signal feature extraction, that must be inherited.")
