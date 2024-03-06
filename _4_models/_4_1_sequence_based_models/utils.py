from typing import Optional
from _4_models.get_pretrained_model import get_pretrained_model


def get_feature_extractor_model(fold_index: int, augmentation_features_logs_folder: Optional[str],
                                augmentation_features_logs_folder_absolute_directory: Optional[str] = None,
                                list_of_valid_architectures: Optional[tuple] = None):
    if augmentation_features_logs_folder is None:
        return None

    feature_extractor_training_model = get_pretrained_model(fold_index, augmentation_features_logs_folder,
                                                            augmentation_features_logs_folder_absolute_directory,
                                                            list_of_valid_architectures)
    return feature_extractor_training_model.extractor_model


def get_pretrained_main_model_components(fold_index: int, main_model_checkpointing_logs_folder: Optional[str],
                                         main_model_checkpointing_logs_folder_absolute_directory: Optional[str] = None,
                                         list_of_valid_architectures: Optional[tuple] = None):

    if main_model_checkpointing_logs_folder is None:
        return None, None, None

    pretrained_main_model = get_pretrained_model(fold_index, main_model_checkpointing_logs_folder,
                                                 main_model_checkpointing_logs_folder_absolute_directory,
                                                 list_of_valid_architectures)

    return pretrained_main_model.intra_element_block, pretrained_main_model.inter_element_block, pretrained_main_model.classification_block


