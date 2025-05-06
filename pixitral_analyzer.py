from vllm import LLM
from vllm.sampling_params import SamplingParams
from logger import get_logger

logger = get_logger(__name__)


def initialize_llm(model_name: str) -> LLM:
    """
    Initialize the PixTral LLM with the specified model name.
    """
    try:
        logger.info(f"Loading model: {model_name}")
        llm = LLM(model=model_name, tokenizer_mode="mistral")
        logger.info("Model loaded successfully.")
        return llm
    except Exception as e:
        logger.exception(f"Failed to initialize LLM: {e}")
        raise


def build_messages(prompt: str, image_url: str) -> list:
    """
    Build the message format expected by PixTral for vision-language input.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]


def run_inference(llm: LLM, messages: list, sampling_params: SamplingParams) -> str:
    """
    Execute inference using the LLM and return the generated clinical summary.
    """
    try:
        logger.info("Running inference...")
        outputs = llm.chat(messages=messages, sampling_params=sampling_params)
        result_text = outputs[0].outputs[0].text.strip()
        logger.info("Inference completed successfully.")
        return result_text
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise