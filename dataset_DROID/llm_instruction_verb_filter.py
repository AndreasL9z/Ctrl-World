import os
from typing import Optional
import warnings



# Default to GPT-4o; can be overridden via model parameter when calling.
_DEFAULT_OPENAI_MODEL = "gpt-4o"
_DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

# --------------------- Prompt Definitions ----------------------
_PROMPTS = {
    "wipe": (
        "You are filtering a robotics instruction dataset. Keep ONLY instructions that describe exactly ONE 'wipe' action and do NOT mention any other actions. "
        "Good examples: 'Wipe the table with the towel', 'Use the towel to wipe the countertop', 'Use the tea towel on the right to wipe the pan'. "
        "Bad examples (should be rejected): 'Move the brown object forward then use the towel to wipe the table'.\n\n"
        "Instruction: \"{instruction}\"\n"
        "Answer strictly with 'YES' (keep) or 'NO' (reject)."
    ),
    "pick_place": (
        "You are filtering a robotics instruction dataset. Keep ONLY instructions that describe exactly ONE pick-and-place action pair and do NOT mention any other actions. "
        "Good examples: 'Pick the lid and put it on the pot', 'Pick up the marker from the cup and place it on the table'. "
        "Bad examples (should be rejected): 'Pick up the lid and close the pot', 'Move the black measuring cup to the right then pick up the glass cup and set it down on the desk'.\n\n"
        "Instruction: \"{instruction}\"\n"
        "Answer strictly with 'YES' (keep) or 'NO' (reject)."
    ),
    "single_action": (
        "You are filtering a robotics instruction dataset. Keep ONLY instructions that describe exactly ONE action and do NOT mention any other actions. "
        "The composite action 'pick and place' (or 'pick and put') counts as ONE action. "
        "Good examples: 'Stack the pillows at the edge of the sofa', 'Pick up the pen and put it in the cup' "
        "Bad examples (should be rejected): 'Slide the tap to the center of the sink and press down the tap handle'.\n\n"
        "Instruction: \"{instruction}\"\n"
        "Answer strictly with 'YES' (keep) or 'NO' (reject)."
    ),
}

# --------------------- LLM Calls ------------------------

import os
from openai import OpenAI

GPT_client = OpenAI(        # Can be omitted to automatically read environment variables
    api_key=os.getenv("OPENAI_API_KEY"),
)

from google import genai                       # pip install -U google-genai
def _call_gpt(prompt: str,
             model: str = "gpt-4o",   # or "gpt-4o-mini"
             temperature: float = 0) -> str | None:
    """
    Call GPT-4o and return the response text. Returns None on error.
    """
    try:
        resp = GPT_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        # v1.x returns a Pydantic object
        return resp.choices[0].message.content.strip()
    except Exception as e:
        warnings.warn(f"OpenAI call failed: {e}")
        return None



import os, warnings
from google import genai                       # pip install -U google-genai
from google.genai import types, errors        # Error classes are here

# Recommended to make Client a singleton to avoid repeated handshakes
_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=_api_key) if _api_key else None

def _call_gemini(prompt: str,
                 model: str = "gemini-1.5-flash",
                 temperature: float = 0.7,
                 timeout_sec: int = 15) -> str | None:
    """Call Gemini (google-genai), returns None on failure."""
    if gemini_client is None:
        warnings.warn("Missing GEMINI_API_KEY / GOOGLE_API_KEY")
        return None

    # 1) Set timeout via GenerateContentConfig in http_options
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        http_options=types.HttpOptions(timeout=timeout_sec * 1000)  # milliseconds!
    )

    try:
        resp = gemini_client.generate_content(
            model=model,
            contents=prompt,     # String will be automatically converted to user-role Content
            config=cfg,
        )
        return resp.text.strip()
    except errors.APIError as e:               # Correct error class location
        warnings.warn(f"Gemini API error: {e}")
        return None
    except Exception as e:
        warnings.warn(f"Gemini call encountered unknown error: {e}")
        return None

# --------------------- Public Interface ------------------------

def instruction_matches_prompt(
    instruction: str,
    *,
    prompt_key: str = "wipe",
    model: str = _DEFAULT_OPENAI_MODEL,
) -> bool:
    """Determine if the instruction meets the requirements using the specified prompt.

    Args:
        instruction: The instruction to evaluate.
        prompt_key: Which prompt to use, can be "wipe" or "pick_place".
        model: GPT model name.

    Returns:
        bool: True if the instruction should be kept, False otherwise.
    """
    instruction = instruction.strip()
    if not instruction:
        return False

    if prompt_key not in _PROMPTS:
        raise ValueError(f"Unknown prompt_key: {prompt_key}. Valid keys: {list(_PROMPTS.keys())}")

    prompt = _PROMPTS[prompt_key].format(instruction=instruction)

    # ----------------- Prefer Gemini -----------------
    llm_answer = _call_gpt(prompt, model=_DEFAULT_OPENAI_MODEL)
    # ----------------- Then GPT ----------------
    if llm_answer is None:
        llm_answer = _call_gemini(prompt, model=_DEFAULT_GEMINI_MODEL)

    if llm_answer is None:
        warnings.warn("⚠️  Neither GPT nor Gemini is available; falling back to heuristics.")

    if llm_answer is not None:
        ans_upper = llm_answer.upper()
        if "YES" in ans_upper:
            return True
        if "NO" in ans_upper:
            return False
        # If answer is unexpected, fall back

    # ---------- Simple Fallback Logic ----------
    # If LLM is unavailable, use heuristics: roughly judge key verb count based on prompt_key
    if prompt_key == "wipe":
        return "wipe" in instruction.lower() and not any(v in instruction.lower() for v in ["pick", "place", "move", "grab", "take"])
    elif prompt_key == "pick_place":
        low = instruction.lower()
        return ("pick" in low or "grab" in low) and ("place" in low or "put" in low) and "wipe" not in low and "move" not in low and "close" not in low
    elif prompt_key == "single_action":
        # Rough heuristics: connectors like "and"/"," /"then" may indicate multiple actions. Allow 'pick and place/put'.
        low = instruction.lower()
        if "pick" in low and ("place" in low or "put" in low):
            # Treat as a single composite action
            return all(word not in low for word in ["wipe", "move", "slide", "stack", "open", "close", "press", "push"]) or True
        # If contains connectors and doesn't belong to allowed pick-place forms, consider as multiple actions
        if any(conn in low for conn in [" and ", " then ", ","]):
            return False
        # Otherwise treat as single action
        return True

    return False


# ------------------ Example Usage -------------------

if __name__ == "__main__":
    demo_instructions = [
        "Wipe the table with the towel",
        "Move the brown object forward then use the towel to wipe the table",
        "Pick the lid and put it on the pot",
        "Slide the tap to the center of the sink and press down the tap handle",
        "Stack the pillows at the edge of the sofa",
    ]

    for key in ["single_action"]: #"wipe", "pick_place"
        print(f"\nPrompt key: {key}")
        for instr in demo_instructions:
            keep = instruction_matches_prompt(instr, prompt_key=key)
            print(f"  [{ 'KEEP' if keep else 'DROP' }] {instr}") 