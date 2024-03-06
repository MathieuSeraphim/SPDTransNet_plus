from torch.nn import Module


class BaseExtractorModel(Module):

    def __init__(self, **init_args):
        super(BaseExtractorModel, self).__init__()
        self.single_output_feature_length = None

    def setup(self, single_output_feature_length: int, **setup_args):
        if type(self) == BaseExtractorModel:
            raise NotImplementedError
        assert single_output_feature_length > 0
        self.single_output_feature_length = single_output_feature_length
        return None

    def forward(self, **inputs):
        raise NotImplementedError
