import argparse
from pixtral_analyzer import initialize_llm, build_messages, run_inference
from vllm.sampling_params import SamplingParams
from logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """
    Parse command-line arguments for the audiogram analyzer.
    """
    parser = argparse.ArgumentParser(
        description="Generate a clinical summary from an audiogram image using PixTral."
    )
    parser.add_argument(
        "--image-url",
        type=str,
        required=True,
        help="Public URL of the audiogram image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "You are a hearing specialist.\n"
            "Given the audiogram image, write a clinical summary describing the degree and type of hearing loss "
            "in both ears and whether it is symmetrical. Use precise medical language. Respond in English."
        ),
        help="Prompt to send to the model"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = "mistralai/Pixtral-12B-2409"
    sampling_params = SamplingParams(max_tokens=1024, temperature=0.6)

    try:
        llm = initialize_llm(model_name)
        messages = build_messages(args.prompt, args.image_url)
        summary = run_inference(llm, messages, sampling_params)
        logger.info("Clinical summary:\n" + summary)
    except Exception:
        logger.error("Execution failed.")


if __name__ == "__main__":
    main()
