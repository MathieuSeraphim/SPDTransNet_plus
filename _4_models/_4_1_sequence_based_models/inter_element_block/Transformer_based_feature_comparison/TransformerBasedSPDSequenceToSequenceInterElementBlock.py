from _4_models._4_1_sequence_based_models.inter_element_block.Transformer_based_feature_comparison.TransformerBasedSequenceToSequenceInterElementBlock import \
    TransformerBasedSequenceToSequenceInterElementBlock
from _4_models._4_1_sequence_based_models.intra_element_block.Transformer_based_feature_extraction.layers.SPDTransformerEncoderLayer import \
    SPDTransformerEncoderLayer


class TransformerBasedSPDSequenceToSequenceInterElementBlock(TransformerBasedSequenceToSequenceInterElementBlock):

    def __init__(self):
        super(TransformerBasedSPDSequenceToSequenceInterElementBlock, self).__init__()
        self.transformer_encoder = SPDTransformerEncoderLayer()


