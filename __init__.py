import logging
from huggingface_hub import snapshot_download
from .zoo import OlmOCR2

logger = logging.getLogger(__name__)


def download_model(model_name, model_path):
    """Downloads the olmOCR-2 model.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name=None, model_path=None, **kwargs):
    """Load an olmOCR-2 model for use with FiftyOne.
    
    Args:
        model_name: Model name (unused, for compatibility)
        model_path: HuggingFace model ID or path to model files
        **kwargs: Additional config parameters (processor_path, custom_prompt, temperature, etc.)
        
    Returns:
        OlmOCR2: Initialized model ready for inference
    """
    if model_path is None:
        model_path = "allenai/olmOCR-2-7B-1025"
    
    return OlmOCR2(model_path=model_path, **kwargs)


def resolve_input(model_name, ctx):
    """Defines properties to collect the model's custom parameters.

    Args:
        model_name: the name of the model
        ctx: an ExecutionContext

    Returns:
        a fiftyone.operators.types.Property
    """
    pass