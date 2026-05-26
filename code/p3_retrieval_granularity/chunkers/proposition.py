import torch
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer

nltk.download('punkt_tab', quiet=True)

_PROMPT = (
    "[INST] Decompose the following passage into a list of short, self-contained propositions. "
    "Each proposition must express exactly one fact and be fully understandable without any surrounding context. "
    "Output one proposition per line. No numbering, no bullet points, no explanation.\n\n"
    "Passage: {passage}\n\n"
    "Propositions: [/INST]"
)

_MIN_PARA_CHARS = 50   # skip paragraphs too short to decompose meaningfully
_MIN_PROP_CHARS = 15   # filter out near-empty lines
_MAX_PROP_CHARS = 300  # filter out lines where the model rambled

_tokenizer = None
_model = None


def _get_model():
    global _tokenizer, _model
    if _model is None:
        name = "mistralai/Mistral-7B-Instruct-v0.2"
        _tokenizer = AutoTokenizer.from_pretrained(name)
        _model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16, device_map="auto"
        )
        _model.eval()
    return _tokenizer, _model


def _decompose(paragraph: str, tokenizer, model) -> list[str]:
    prompt = _PROMPT.format(passage=paragraph)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    props = [
        line.strip()
        for line in text.splitlines()
        if _MIN_PROP_CHARS <= len(line.strip()) <= _MAX_PROP_CHARS
    ]
    return props if props else [paragraph]


def chunk(text: str, tokenizer=None, model=None) -> list[str]:
    """
    Decompose a document into atomic propositions using Mistral-7B-Instruct.

    Each paragraph is passed to the model with a decomposition prompt.
    Short paragraphs (< 50 chars) are skipped. If the model produces no
    valid output for a paragraph, the paragraph is returned as-is.

    Args:
        text: raw document text
        tokenizer: optional pre-loaded tokenizer (avoids reloading)
        model: optional pre-loaded model (avoids reloading)

    Returns:
        list of atomic proposition strings
    """
    t, m = (tokenizer, model) if (tokenizer is not None and model is not None) else _get_model()

    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) >= _MIN_PARA_CHARS]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    all_props = []
    for para in paragraphs:
        all_props.extend(_decompose(para, t, m))
    return all_props
