"""CNN Visual Learning: the Streamlit deployment entry point."""
import streamlit as st
from framework_loader import load_module


PAGES = {
    "Learn": ["Home", "Introduction", "Guided Tutorials", "Photo Lab", "Animation Lab"],
    "Experiment": ["Light Curve Lab", "Process Sample Images", "Upload Your Own Image", "Custom Filter Builder", "Advanced Visualizations"],
    "Build & Train": ["CNN from Scratch", "CNN with Libraries", "Training Visualization", "Layer-by-Layer Training", "Architecture Builder", "Classification Playground", "Comparison"],
}
DESCRIPTIONS = {
    "Light Curve Lab": "Explore stellar brightness with animated 1D convolution and pooling.",
    "Photo Lab": "Watch real photographs change through pooling, convolution, and activation.",
    "Introduction": "Follow an image from pixels to features and predictions.",
    "Guided Tutorials": "Build intuition with step-by-step CNN lessons.",
    "Animation Lab": "Watch windows move, compare pooling, and explore activations.",
    "Process Sample Images": "Try edge, blur, and sharpening filters on built-in patterns.",
    "Upload Your Own Image": "Explore the features hidden in your own images.",
    "Custom Filter Builder": "Design a kernel and see exactly what it detects.",
    "Advanced Visualizations": "Explore receptive fields, feature highlighting, and backpropagation.",
    "CNN from Scratch": "Trace and train a small CNN built with NumPy.",
    "CNN with Libraries": "Inspect the same ideas in TensorFlow and PyTorch.",
    "Training Visualization": "Watch loss, accuracy, and learned filters change during training.",
    "Layer-by-Layer Training": "Explore how intermediate activations evolve on a synthetic dataset.",
    "Architecture Builder": "Assemble layers and export model code.",
    "Classification Playground": "Inspect predictions, activation maps, and feature maps.",
    "Comparison": "Compare operations and illustrative timings across implementations.",
}
EXTRA_PAGES = {
    "Light Curve Lab": ("light_curve_lab", "render"),
    "Photo Lab": ("photo_explorer", "render"),
    "Layer-by-Layer Training": ("training_visualization", "visualize_training_progress"),
    "Guided Tutorials": ("guided_tutorials", "guided_tutorials"),
    "Animation Lab": ("operations_explorer", "render"),
    "Advanced Visualizations": ("advanced_visualizations", "advanced_visualizations"),
    "Architecture Builder": ("architecture_builder", "architecture_builder"),
    "Classification Playground": ("classification_playground", "image_classification_playground"),
}
MODEL_PAGES = {"CNN from Scratch", "CNN with Libraries", "Training Visualization", "Comparison"}


def open_page(page):
    st.session_state.page = page


def home():
    st.markdown('<div class="hero"><span class="eyebrow">AN INTERACTIVE LEARNING LAB</span><h1>See how a CNN sees.</h1><p>Turn pixels into patterns. Move through each operation, build your intuition, and train a network of your own.</p></div>', unsafe_allow_html=True)
    st.subheader("Start with one small experiment")
    for col, (page, label) in zip(st.columns(3), [
        ("Photo Lab", "01 · Watch a photo change"),
        ("Process Sample Images", "02 · Try a filter"),
        ("CNN from Scratch", "03 · Train a CNN"),
    ]):
        with col, st.container(border=True):
            st.markdown(f"### {label}")
            st.write(DESCRIPTIONS[page])
            st.button("Open activity →", key=f"home_{page}", on_click=open_page, args=(page,), width="stretch")
    st.subheader("Your learning path")
    st.write("Pixels → Convolution → Activation → Pooling → Classification")
    for col, (group, pages) in zip(st.columns(3), PAGES.items()):
        with col:
            st.markdown(f"### {group}")
            for page in pages:
                if page != "Home":
                    with st.expander(page):
                        st.write(DESCRIPTIONS[page])
                        st.button("Explore →", key=f"directory_{page}", on_click=open_page, args=(page,))
    st.info("New to CNNs? Start with the Photo Lab. The basic activities use built-in images and need no dataset downloads.")


def main():
    st.set_page_config(page_title="CNN Visual Learning", page_icon="🧠", layout="wide")
    st.markdown('''<style>
    .block-container {max-width: 1280px; padding-top: 2rem;}
    .hero {padding: 2.5rem; border-radius: 24px; background: linear-gradient(120deg,#112c3b,#145a62); color: #fff; margin-bottom: 2rem;}
    .hero h1 {color: #fff; font-size: clamp(2rem,5vw,3.6rem); letter-spacing: -.04em; padding: .6rem 0;}
    .hero p {max-width: 660px; font-size: 1.1rem; color: #e1f4f2;}
    .eyebrow {font-size: .75rem; letter-spacing: .16em; color: #9ce1d5;}
    @media (max-width: 900px) {
      [data-testid="stHorizontalBlock"] {flex-direction: column;}
      [data-testid="stColumn"] {width: 100%; flex: 1 1 100%;}
      .hero {padding: 1.5rem;}
    }
    </style>''', unsafe_allow_html=True)
    with st.sidebar:
        st.title("🧠 CNN Visual Learning")
        st.caption("Explore. Experiment. Understand.")
        all_pages = [page for pages in PAGES.values() for page in pages]
        page = st.selectbox("Activity", all_pages, key="page")
        st.divider()
        st.caption("LEARNING PATH")
        st.caption("Learn the concepts, experiment with images, then build and train.")
        settings = dict(learning_rate=0.01, batch_size=32, num_epochs=3, conv_filters=16)
        if page in MODEL_PAGES:
            st.divider()
            with st.expander("Training settings", expanded=True):
                settings["learning_rate"] = st.slider("Learning rate", 0.001, 0.1, 0.01, 0.001)
                settings["batch_size"] = st.select_slider("Batch size", [16, 32, 64, 128], value=32)
                settings["num_epochs"] = st.slider("Epochs", 1, 10, 3)
                settings["conv_filters"] = st.slider("Convolution filters", 4, 32, 16, 4)
                if st.button("Reset models"):
                    st.session_state.pop("cnn_models", None)
                    st.success("Models reset for this session.")
                st.caption("Weights persist while you explore. Changing model settings creates a fresh model.")
    try:
        if page == "Home":
            home()
        else:
            import matplotlib.pyplot as plt
            plt.style.use("dark_background")
            plt.rcParams.update({"figure.facecolor": "#0b1220", "axes.facecolor": "#142033", "savefig.facecolor": "#0b1220"})
            st.caption(f"CNN VISUAL LEARNING / {next(group for group, pages in PAGES.items() if page in pages).upper()}")
            st.write(DESCRIPTIONS[page])
            if page in EXTRA_PAGES:
                module, function = EXTRA_PAGES[page]
                getattr(load_module(module), function)()
            elif page in MODEL_PAGES:
                load_module("core_pages").render(page, **settings)
            else:
                load_module("complete_cnn_copy").appmode(page).call_app()
    finally:
        # Streamlit has captured each figure; release server resources on reruns.
        import sys
        if "matplotlib.pyplot" in sys.modules:
            sys.modules["matplotlib.pyplot"].close("all")
    st.divider()
    st.caption("CNN Visual Learning · Created by Shivaji Chaulagain · MIT License")


if __name__ == "__main__":
    main()
