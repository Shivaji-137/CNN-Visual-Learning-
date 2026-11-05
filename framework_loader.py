"""Import optional deep learning backends only when a page needs them."""
import importlib


def load_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        import streamlit as st
        st.error(f"This activity needs an additional library: {exc.name}.")
        st.info("Install the full requirements.txt dependencies, then restart the app. Image demos and the animation lab work without deep learning frameworks.")
        st.stop()


class LazyModule:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, attribute):
        return getattr(load_module(self.name), attribute)
