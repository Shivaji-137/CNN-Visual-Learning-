from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pytest
from streamlit.testing.v1 import AppTest
from full_apps import PAGES


@pytest.mark.parametrize('page', [page for pages in PAGES.values() for page in pages])
def test_pages_open(page):
    app = AppTest.from_file(ROOT / 'full_apps.py', default_timeout=90).run()
    app.selectbox(key='page').select(page).run()
    assert not app.exception, [exc.message for exc in app.exception]


@pytest.mark.parametrize('operation', ['Max pooling', 'Average pooling', 'Convolution', 'Activation', 'Flatten'])
def test_animation_controls(operation):
    app = AppTest.from_file(ROOT / 'full_apps.py', default_timeout=60).run()
    app.selectbox(key='page').select('Animation Lab').run()
    next(widget for widget in app.selectbox if widget.label == 'Operation').select(operation).run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 1


def test_no_filters_shows_guidance():
    app = AppTest.from_file(ROOT / 'full_apps.py', default_timeout=60).run()
    app.selectbox(key='page').select('Process Sample Images').run()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any('at least one filter' in item.value for item in app.info)


def test_home_activity_link_and_alternate_entrypoint():
    app = AppTest.from_file(ROOT / 'main.py').run()
    app.button(key='home_Photo Lab').click().run()
    assert app.selectbox(key='page').value == 'Photo Lab'
    assert not app.exception


def test_torch_live_training_and_weights_survive_rerun():
    pytest.importorskip('torch')
    app = AppTest.from_file(ROOT / 'full_apps.py', default_timeout=90).run()
    app.selectbox(key='page').select('Training Visualization').run()
    next(widget for widget in app.radio if widget.label == 'Select Implementation').set_value('PyTorch').run()
    next(widget for widget in app.slider if widget.label == 'Epochs').set_value(1).run()
    next(widget for widget in app.button if widget.label == 'Start Training Visualization').click().run()
    assert not app.exception
    assert any('Training completed' in item.value for item in app.success)
    before = app.session_state['cnn_models'][('Training Visualization','pytorch')][1]
    app.run()
    assert app.session_state['cnn_models'][('Training Visualization','pytorch')][1] is before


def test_sample_image_numeric_feature_animations():
    app = AppTest.from_file(ROOT / 'full_apps.py', default_timeout=60).run()
    app.selectbox(key='page').select('Process Sample Images').run()
    app.multiselect[0].set_value(['Edge Detection (Horizontal)']).run()
    prefix = 'feature_Edge Detection (Horizontal)'
    app.checkbox(key=f'{prefix}_enabled').check().run()
    app.radio(key=f'{prefix}_view').set_value('Numbers close-up').run()
    for operation in ['Max pooling','Average pooling','Convolution','Activation','Flatten']:
        app.selectbox(key=f'{prefix}_operation').select(operation).run()
        assert not app.exception, [exc.message for exc in app.exception]
        assert len(app.get('plotly_chart')) == 4  # Three surface plots plus numeric animation.
    app.selectbox(key=f'{prefix}_operation').select('Activation').run()
    next(w for w in app.selectbox if w.label == 'Select activation function').select('Leaky ReLU').run()
    assert not app.exception
