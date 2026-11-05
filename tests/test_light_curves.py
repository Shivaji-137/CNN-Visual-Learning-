from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from light_curve_lab import validate_curve, operate_curve, simulated_curve, curve_animation, KERNELS


def test_sorting_preserves_measurement_pairs_and_rejects_invalid_values():
    time,flux=validate_curve([3,1,2],[30,10,20])
    np.testing.assert_array_equal(time,[1,2,3]); np.testing.assert_array_equal(flux,[10,20,30])
    for t,f in [([1,1,2],[2,3,4]),([1,2,3],[1,np.nan,3]),([1,2],[1,2])]:
        with pytest.raises(ValueError): validate_curve(t,f)


@pytest.mark.parametrize('operation,expected',[('Max pooling',[4,8]),('Average pooling',[2.5,6.5]),('Minimum pooling',[1,5])])
def test_pooling_values_timestamps_and_trailing_windows(operation,expected):
    result=operate_curve(np.arange(9),np.arange(1,10),operation,window=4,stride=4)
    np.testing.assert_array_equal(result['output'],expected)
    np.testing.assert_array_equal(result['output_time'],[1.5,5.5])


def test_gap_crossing_windows_are_skipped_without_interpolation():
    result=operate_curve([0,1,2,20,21,22],[1,2,3,4,5,6],'Average pooling',window=3,stride=1)
    assert result['skipped']==2
    np.testing.assert_array_equal(result['output'],[2,5])
    np.testing.assert_array_equal(result['output_time'],[1,21])


def test_cnn_correlation_and_centered_activation():
    result=operate_curve(np.arange(7),np.arange(7),'Convolution',stride=1,kernel=[1,0,-1])
    np.testing.assert_array_equal(result['output'],-2*np.ones(5))
    result=operate_curve([0,1,2],[-1,0,2],'Activation')
    np.testing.assert_array_equal(result['output'],[0,0,2])


def test_observed_data_provenance_and_animation_final_frame():
    data=pd.read_csv(Path(__file__).resolve().parents[1]/'data/kepler8_q0.csv')
    assert len(data)>400 and (data.sap_quality==0).all()
    time,flux=validate_curve(data.time_bkjd,data.pdcsap_flux)
    assert 120<time[0]<121 and 129<time[-1]<131
    result=operate_curve(time,flux/np.median(flux),'Convolution',kernel=KERNELS['Smooth (5-point mean)'],stride=1)
    figure=curve_animation(result,'Convolution')
    assert len(figure.frames)==80
    np.testing.assert_array_equal(figure.frames[-1].data[-1].y,result['output'])


def test_light_curve_page_operations_and_simulations():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'full_apps.py',default_timeout=60).run()
    app.selectbox(key='page').select('Light Curve Lab').run()
    assert not app.exception
    for operation in ['Max pooling','Average pooling','Minimum pooling','Activation','Convolution']:
        next(w for w in app.selectbox if w.label=='Light-curve operation').select(operation).run()
        assert not app.exception
        assert len(app.get('plotly_chart'))==1
    for signal in ['Simulated transit','Simulated variable star','Simulated flare']:
        time,flux=simulated_curve(signal)
        assert len(time)==512 and np.isfinite(flux).all()
        next(w for w in app.selectbox if w.label=='Light curve').select(signal).run()
        assert not app.exception
