import logging
import sys
import io
import warnings
from contextlib import contextmanager
from typing import Union

from PIL import Image
import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """Attached is one page of a document that you must process. Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to HTML.
If there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)
Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."""

@contextmanager
def suppress_output():
    """Suppress stdout, stderr, warnings, and transformers logging."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    # Suppress transformers logging
    transformers_logger = logging.getLogger("transformers")
    old_transformers_level = transformers_logger.level
    transformers_logger.setLevel(logging.ERROR)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        transformers_logger.setLevel(old_transformers_level)



def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OlmOCR2(Model, SamplesMixin):
    """FiftyOne model for olmOCR-2 vision-language OCR tasks.
    
    Advanced OCR model that extracts text from documents using Qwen2.5-VL architecture.
    Returns markdown output with YAML front matter containing metadata about the document.
    
    Automatically selects optimal dtype based on hardware:
    - bfloat16 for CUDA devices with compute capability 8.0+ (Ampere and newer)
    - float16 for older CUDA devices
    - float32 for CPU/MPS devices
    
    Args:
        model_path: HuggingFace model ID or local path (default: "allenai/olmOCR-2-7B-1025")
        processor_path: HuggingFace processor ID (default: "Qwen/Qwen2.5-VL-7B-Instruct")
        custom_prompt: Custom prompt for OCR task (optional)
        max_new_tokens: Maximum tokens to generate (default: 4096)
        temperature: Temperature for sampling (default: 0.1)
        torch_dtype: Override automatic dtype selection
    """
    
    def __init__(
        self,
        model_path: str = "allenai/olmOCR-2-7B-1025",
        processor_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        custom_prompt: str = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        torch_dtype: torch.dtype = None,
        **kwargs
    ):
        SamplesMixin.__init__(self) 
        self.model_path = model_path
        self.processor_path = processor_path
        self._custom_prompt = custom_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Device setup
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Dtype selection
        if torch_dtype is not None:
            self.dtype = torch_dtype
        elif self.device == "cuda":
            capability = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
            logger.info(f"Using {self.dtype} dtype (compute capability {capability[0]}.{capability[1]})")
        else:
            self.dtype = torch.float32
            logger.info(f"Using float32 dtype for {self.device}")
        
        # Load model and processor
        logger.info(f"Loading olmOCR-2 from {model_path}")
        
        self.processor = AutoProcessor.from_pretrained(processor_path)
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype
        ).eval()
        
        self.model.to(self.device)
        
        logger.info("olmOCR-2 model loaded successfully")
    
    @property
    def media_type(self):
        """The media type processed by this model."""
        return "image"
    
    def _predict(self, image: Image.Image, sample) -> str:
        """Process image through olmOCR-2.
        
        Args:
            image: PIL Image to process
            sample: FiftyOne sample (has filepath attribute)
        
        Returns:
            str: Extracted text from the document with YAML front matter
        """
        # Use custom prompt if provided, otherwise use default
        prompt = self._custom_prompt if self._custom_prompt else DEFAULT_PROMPT
        
        # Prepare messages in the chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        
        # Run inference with suppressed output
        with suppress_output():
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Generate output
            output = self.model.generate(
                **inputs,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=1,
                do_sample=True,
            )
            
            # Decode only the generated tokens (excluding input prompt)
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_length:]
            
            # Decode to text
            text_output = self.processor.tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True
            )
            
            result = text_output[0]
        
        return result
    
    def predict(self, image, sample=None):
        """Process an image with olmOCR-2.
        
        Args:
            image: PIL Image or numpy array to process
            sample: FiftyOne sample containing the image filepath
        
        Returns:
            str: Extracted text from the document with YAML front matter
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)