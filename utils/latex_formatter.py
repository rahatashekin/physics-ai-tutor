"""
Utility: LaTeX Delimiter Formatter
====================================
GPT এর output এ LaTeX delimiter normalize করে Streamlit render এর জন্য।
"""

import re


def format_latex_for_streamlit(text: str) -> str:
    """
    GPT এর output এ \\(...\\) এবং \\[...\\] delimiter গুলো
    Streamlit এর $...$ এবং $$...$$ format এ convert করে।
    """
    # Display math: \[...\] → $$...$$
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    # Inline math: \(...\) → $...$
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    return text
