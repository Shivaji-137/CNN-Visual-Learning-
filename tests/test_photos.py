import numpy as np
import pytest
from scipy.signal import correlate2d
from photo_explorer import photo_steps, photo_animation, sample_photos
from operations_explorer import window_steps


def test_real_photographs_are_local_rgb_images():
    photos=sample_photos()
    assert len(photos)==2
    for photo in photos:
        assert photo['image'].mode=='RGB'
        assert max(photo['image'].size)<=128
        assert np.unique(np.asarray(photo['image']).reshape(-1,3),axis=0).shape[0]>100


@pytest.mark.parametrize('operation',['Max pooling','Average pooling','Convolution','Activation'])
def test_photo_checkpoints_end_with_complete_correct_image(operation):
    source=np.arange(400,dtype=float).reshape(20,20)/400-.5
    kernel=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    output,steps=photo_steps(source,operation,kernel,2,checkpoints=8)
    assert len(steps)==8
    np.testing.assert_array_equal(steps[-1]['output'],output)
    assert np.count_nonzero(np.isfinite(steps[0]['output']))==1
    if operation=='Convolution':
        np.testing.assert_allclose(output,correlate2d(source,kernel,mode='valid'),atol=1e-15)
    elif operation=='Activation':
        np.testing.assert_array_equal(output,np.maximum(source,0))
    else:
        expected,_=window_steps(source,operation,2,2)
        np.testing.assert_allclose(output,expected)
    figure=photo_animation(source,output,steps,operation,np.repeat(source[:,:,None]+.5,3,axis=2))
    assert len(figure.data)==3 and len(figure.frames)==8
    assert figure.frames[-1].traces==(2,)
    assert len(figure.frames[-1].layout.shapes)==3


def test_photo_lab_controls_and_sample_full_image_view():
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    entry=Path(__file__).resolve().parents[1]/'full_apps.py'
    app=AppTest.from_file(entry,default_timeout=60).run()
    app.selectbox(key='page').select('Photo Lab').run()
    for operation in ['Max pooling','Average pooling','Convolution','Activation']:
        next(w for w in app.selectbox if w.label=='Photo operation').select(operation).run()
        assert not app.exception, [e.message for e in app.exception]
        assert len(app.get('plotly_chart'))==1
    app.selectbox(key='page').select('Process Sample Images').run()
    assert next(w for w in app.selectbox if w.label=='Select a sample image').value=='Garden photograph'
    app.multiselect[0].set_value(['Edge Detection (Horizontal)']).run()
    app.checkbox(key='feature_Edge Detection (Horizontal)_enabled').check().run()
    assert not app.exception
    assert app.radio(key='feature_Edge Detection (Horizontal)_view').value=='Whole image'
    assert len(app.get('plotly_chart'))==4
