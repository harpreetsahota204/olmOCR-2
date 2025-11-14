import logging
import sys
import io
import warnings
from contextlib import contextmanager
from typing import Union
import re

from PIL import Image
import numpy as np
import torch
import yaml

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
    Automatically parses YAML front matter and returns multiple fields in one inference pass:
    - text: Main OCR content (markdown)
    - primary_language: Detected language (Classification)
    - is_rotation_valid: Whether orientation is correct (Classification)
    - rotation_correction: Suggested rotation degrees (numeric)
    - is_table: Whether document contains tables (Classification)
    - is_diagram: Whether document contains diagrams (Classification)
    
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
    
    Example:
        >>> model = OlmOCR2()
        >>> dataset.apply_model(model, label_field="ocr")
        >>> # Creates: sample.ocr_text, sample.ocr_primary_language, 
        >>> #          sample.ocr_is_table, sample.ocr_is_diagram, etc.
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
    
    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        """Parse YAML front matter from markdown text.
        
        Args:
            text: Markdown text with optional YAML front matter
            
        Returns:
            tuple: (metadata_dict, content_without_frontmatter)
        """
        # Match YAML front matter pattern: ---\n...content...\n---
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.match(frontmatter_pattern, text, re.DOTALL)
        
        if match:
            try:
                yaml_content = match.group(1)
                main_content = match.group(2)
                metadata = yaml.safe_load(yaml_content) or {}
                return metadata, main_content
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse YAML front matter: {e}")
                return {}, text
        
        # No front matter found
        return {}, text
    
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
            dict: Dictionary with multiple fields:
                - text: Main OCR content
                - primary_language: Classification with detected language
                - is_rotation_valid: Classification (valid/invalid)
                - rotation_correction: Numeric rotation value
                - is_table: Classification (true/false)
                - is_diagram: Classification (true/false)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Get raw model output
        raw_output = self._predict(image, sample)
        
        # Parse front matter
        metadata, text_content = self._parse_frontmatter(raw_output)
        
        # Build result dictionary with multiple fields
        result = {
            "text": text_content.strip()
        }
        
        # Add metadata fields as Classifications
        if "primary_language" in metadata:
            result["primary_language"] = fo.Classification(
                label=str(metadata["primary_language"])
            )
        
        if "is_rotation_valid" in metadata:
            # Convert boolean to string label
            label = str(metadata["is_rotation_valid"]).lower()
            result["is_rotation_valid"] = fo.Classification(label=label)
        
        if "rotation_correction" in metadata:
            # Store as plain numeric value
            result["rotation_correction"] = metadata["rotation_correction"]
        
        if "is_table" in metadata:
            # Convert boolean to string label
            label = str(metadata["is_table"]).lower()
            result["is_table"] = fo.Classification(label=label)
        
        if "is_diagram" in metadata:
            # Convert boolean to string label
            label = str(metadata["is_diagram"]).lower()
            result["is_diagram"] = fo.Classification(label=label)
        
        return result