"""Minimal stub of the Streamlit library for testing purposes.
Only the symbols used in the codebase are provided. This avoids pulling the
full Streamlit dependency in the constrained execution environment.
"""

def error(msg: str):
    """Placeholder for ``st.error`` – simply prints the message.
    In the real Streamlit UI this would display a red error box.
    """
    print(f"[streamlit error] {msg}")

# Export ``error`` at module level so ``import streamlit as st`` provides
# ``st.error`` directly.
__all__ = ["error"]
