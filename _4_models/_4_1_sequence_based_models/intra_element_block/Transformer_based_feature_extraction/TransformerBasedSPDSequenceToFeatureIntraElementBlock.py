from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.TransformerBasedSequenceToFeatureIntraElementBlock import \
    TransformerBasedSequenceToFeatureIntraElementBlock
from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.layers.SPDTransformerEncoderLayer import \
    SPDTransformerEncoderLayer


class TransformerBasedSPDSequenceToFeatureIntraElementBlock(TransformerBasedSequenceToFeatureIntraElementBlock):

    def __init__(self):
        super(TransformerBasedSPDSequenceToFeatureIntraElementBlock, self).__init__()
        self.transformer_encoder = SPDTransformerEncoderLayer()

