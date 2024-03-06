import copy
from typing import List, Optional
import torch
from torch.nn import Linear
from _4_models._4_1_sequence_based_models.classification_block.BaseClassificationBlock import BaseClassificationBlock
from _4_models._4_1_sequence_based_models.classification_block.classification_from_sequence_central_feature.CentralGroupOfFeaturesInSequenceClassificationBlock import \
    CentralGroupOfFeaturesInSequenceClassificationBlock
from _4_models._4_1_sequence_based_models.classification_block.classification_from_sequence_central_feature.layers.FullyConnectedLayer import \
    FullyConnectedLayer


class FinetunedCentralGroupOfFeaturesInSequenceClassificationBlock(CentralGroupOfFeaturesInSequenceClassificationBlock):

    def __init__(self):
        self.COMPATIBLE_CLASSIFICATION_BLOCKS = (
            CentralGroupOfFeaturesInSequenceClassificationBlock,
            FinetunedCentralGroupOfFeaturesInSequenceClassificationBlock
        )
        super(FinetunedCentralGroupOfFeaturesInSequenceClassificationBlock, self).__init__()

    def setup(self, sequence_length: int, vector_size: int, number_of_classes: int, relevant_feature_indices: List[int],
              fully_connected_intermediary_dimension: int, fully_connected_dropout_rate: float,
              pretrained_classification_block: Optional[CentralGroupOfFeaturesInSequenceClassificationBlock] = None):
        super(FinetunedCentralGroupOfFeaturesInSequenceClassificationBlock, self).setup(
            sequence_length, vector_size, number_of_classes, relevant_feature_indices,
            fully_connected_intermediary_dimension, fully_connected_dropout_rate)

        if pretrained_classification_block is None:
            return

        # Checking that we're using the same parameters
        assert isinstance(pretrained_classification_block, self.COMPATIBLE_CLASSIFICATION_BLOCKS)
        assert pretrained_classification_block.sequence_length == self.sequence_length
        assert pretrained_classification_block.vector_size == self.vector_size
        assert pretrained_classification_block.number_of_classes == self.number_of_classes
        assert pretrained_classification_block.relevant_feature_indices == self.relevant_feature_indices
        assert pretrained_classification_block.classification_layer.in_features == fully_connected_intermediary_dimension
        assert pretrained_classification_block.first_fc_layer.dropout_layer.p == fully_connected_dropout_rate

        self.first_fc_layer = copy.deepcopy(pretrained_classification_block.first_fc_layer)
        self.second_fc_layer = copy.deepcopy(pretrained_classification_block.second_fc_layer)
        self.classification_layer = copy.deepcopy(pretrained_classification_block.classification_layer)








