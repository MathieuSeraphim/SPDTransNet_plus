from _4_models._4_2_signal_feature_extractor_models.features_to_classification_block.BaseFeaturesToClassificationBlock import \
    BaseFeaturesToClassificationBlock
from torch.nn import Linear


class MultichannelFeatureExtractorPretrainingBlock(BaseFeaturesToClassificationBlock):

    def __init__(self):
        super(MultichannelFeatureExtractorPretrainingBlock, self).__init__()
        self.__setup_done_flag = False

        self.features_size = None
        self.classification_layer = None

    def setup(self, first_feature_dimension: int, second_feature_dimension: int, third_feature_dimension: int,
              number_of_classes: int):
        assert not self.__setup_done_flag

        self.features_size = first_feature_dimension * second_feature_dimension * third_feature_dimension
        self.classification_layer = Linear(self.features_size, number_of_classes)

        self.__setup_done_flag = True

    def forward(self, features):
        assert self.__setup_done_flag

        batch_size = features.shape[0]
        assert len(features.shape) == 4
        features = features.view(batch_size, self.features_size)

        output = self.classification_layer(features)
        return output



