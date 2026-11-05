"""Astronomical light curves: observed data, simulations, and animated 1D CNN operations."""
from pathlib import Path
import numpy as np

KERNELS = {
    'Smooth (5-point mean)': np.ones(5)/5,
    'Brightness change': np.array([-1., 0., 1.])/2,
    'Dip contrast': np.array([.25, .25, -1., .25, .25]),
}


def validate_curve(time, flux):
    time, flux = np.asarray(time, dtype=float), np.asarray(flux, dtype=float)
    if time.ndim != 1 or flux.ndim != 1 or len(time) != len(flux) or len(time) < 3:
        raise ValueError('Provide at least three matching time and brightness samples.')
    if not np.isfinite(time).all() or not np.isfinite(flux).all():
        raise ValueError('Time and brightness must contain only finite numeric values.')
    order = np.argsort(time, kind='stable')
    time, flux = time[order], flux[order]
    if np.any(np.diff(time) <= 0):
        raise ValueError('Time values must be unique. Combine repeated measurements before uploading.')
    return time, flux


def simulated_curve(name, count=512):
    time = np.linspace(0, 12, count)
    rng = np.random.default_rng(42)
    if name == 'Simulated transit':
        phase = (time-1.5+1.75)%3.5-1.75
        flux = 1-.018*np.exp(-.5*(phase/.12)**4)
    elif name == 'Simulated variable star':
        flux = 1+.04*np.sin(2*np.pi*time/2.4)+.01*np.sin(4*np.pi*time/2.4)
    else:
        delta = time-5
        flux = 1+.07*np.exp(-np.maximum(delta,0)/.45)*np.exp(-np.minimum(delta,0)**2/.015)
    return time, flux+rng.normal(0,.001,count)


def operate_curve(time, flux, operation, window=4, stride=4, kernel=None, activation='ReLU', skip_gaps=True):
    from operations_explorer import activation_values
    time, flux = validate_curve(time,flux)
    if operation == 'Activation':
        window, stride = 1, 1
    elif operation == 'Convolution':
        kernel = np.asarray(kernel,dtype=float)
        if kernel.ndim != 1 or not len(kernel) or not np.isfinite(kernel).all():
            raise ValueError('Use a finite, nonempty 1D kernel.')
        window = len(kernel)
    elif operation not in {'Max pooling','Average pooling','Minimum pooling'}:
        raise ValueError('Unknown light-curve operation.')
    if window < 1 or stride < 1 or window > len(flux):
        raise ValueError('Window and stride must be positive, and the window must fit the curve.')
    starts = np.arange(0,len(flux)-window+1,stride)
    indices = starts[:,None]+np.arange(window)
    regions = flux[indices]
    allowed = np.ones(len(starts),dtype=bool)
    if skip_gaps and window>1:
        cadence = np.median(np.diff(time))
        allowed = np.max(np.diff(time[indices],axis=1),axis=1) <= 3*cadence
    starts, indices, regions = starts[allowed], indices[allowed], regions[allowed]
    if not len(starts):
        raise ValueError('No complete windows remain. Reduce the window size or review the time gaps.')
    if operation == 'Max pooling':
        values = regions.max(axis=1)
    elif operation == 'Minimum pooling':
        values = regions.min(axis=1)
    elif operation == 'Average pooling':
        values = regions.mean(axis=1)
    elif operation == 'Convolution':
        values = regions@kernel
    else:
        values = activation_values(regions[:,0],activation)
    return dict(time=time,flux=flux,output_time=time[indices].mean(axis=1),output=values,
                starts=starts,window=window,stride=stride,skipped=int((~allowed).sum()))


def curve_animation(result, operation, time_label='Time (days)', speed=220):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    time,flux,values = result['time'],result['flux'],result['output']
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.15,
                     subplot_titles=('Input light curve · active window','Output · computed values'))
    fig.add_trace(go.Scatter(x=time,y=flux,mode='lines',line=dict(color='#7dd3fc',width=1.5),name='Input brightness'),1,1)
    fig.add_trace(go.Scatter(x=[],y=[],mode='markers+lines',line=dict(color='#fbbf24'),marker=dict(size=8),name='Window'),1,1)
    fig.add_trace(go.Scatter(x=[],y=[],mode='markers',marker=dict(size=12,color='#5eead4',symbol='diamond'),name='Selected extremum'),1,1)
    fig.add_trace(go.Scatter(x=result['output_time'],y=np.full(len(values),np.nan),mode='lines+markers',
                            line=dict(color='#c4b5fd'),marker=dict(size=5),name='Output'),2,1)
    positions=np.unique(np.linspace(0,len(values)-1,min(80,len(values)),dtype=int))

    def snapshot(position):
        start=result['starts'][position]; size=result['window']
        tx,fy=time[start:start+size],flux[start:start+size]
        if operation in {'Max pooling','Minimum pooling'}:
            target=fy.max() if operation=='Max pooling' else fy.min()
            selected=fy==target
            sx,sy=tx[selected],fy[selected]
        else:
            sx,sy=[],[]
        revealed=np.full(len(values),np.nan); revealed[:position+1]=values[:position+1]
        detail=', '.join(f'{v:.5g}' for v in fy) if size<=5 else f'{size} brightness samples'
        title=dict(text=f'{operation} · output {position+1}/{len(values)}<br><sup>[{detail}] → {values[position]:.6g}</sup>',x=.5,xanchor='center',font=dict(size=16))
        shapes=[dict(type='rect',xref='x',yref='y domain',x0=tx[0],x1=tx[-1],y0=0,y1=1,
                     fillcolor='rgba(251,191,36,.12)',line=dict(color='#fbbf24',width=2))]
        data=[go.Scatter(x=tx,y=fy),go.Scatter(x=sx,y=sy),go.Scatter(x=result['output_time'],y=revealed)]
        return data,title,shapes

    frames=[]
    for i,position in enumerate(positions):
        data,title,shapes=snapshot(position)
        frames.append(go.Frame(name=str(i),traces=[1,2,3],data=data,layout=go.Layout(title=title,shapes=shapes)))
    fig.frames=frames
    data,title,shapes=snapshot(0)
    for trace,update in zip(fig.data[1:],data):
        trace.update(x=update.x,y=update.y)
    instant=dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))
    fig.update_layout(template='plotly_dark',height=680,paper_bgcolor='rgba(0,0,0,0)',title=title,shapes=shapes,
                      margin=dict(t=110,b=145,l=65,r=25),showlegend=False,
                      updatemenus=[dict(type='buttons',direction='left',x=0,y=-.16,showactive=False,bgcolor='#142033',buttons=[
                          dict(label='▶ Play',method='animate',args=[None,dict(fromcurrent=True,frame=dict(duration=speed,redraw=True),transition=dict(duration=0))]),
                          dict(label='Ⅱ Pause',method='animate',args=[[None],instant]),
                          dict(label='↺ Restart',method='animate',args=[['0'],instant])])],
                      sliders=[dict(x=.03,len=.94,y=-.3,currentvalue=dict(prefix='Checkpoint '),steps=[
                          dict(label=str(int(position)+1),method='animate',args=[[str(i)],instant]) for i,position in enumerate(positions)])])
    fig.update_xaxes(title_text=time_label,row=2,col=1)
    fig.update_yaxes(title_text='Input brightness',row=1,col=1)
    low,high=float(values.min()),float(values.max()); padding=max((high-low)*.12,abs(high)*.005,1e-9)
    fig.update_yaxes(title_text='Output value',range=[low-padding,high+padding],row=2,col=1)
    fig.update_xaxes(range=[float(time[0]),float(time[-1])])
    return fig


def render():
    import pandas as pd
    import streamlit as st
    st.title('Light Curve Lab')
    st.write('Watch a window move across stellar brightness measurements. See which samples contribute, which extrema survive pooling, and how the output changes over time.')
    observed=Path(__file__).parent/'data'/'kepler8_q0.csv'
    choices=(['Observed Kepler-8 (NASA/MAST)'] if observed.exists() else [])+['Simulated transit','Simulated variable star','Simulated flare','Upload CSV']
    selected=st.selectbox('Light curve',choices)
    time_label='Time (days)'
    if selected=='Upload CSV':
        upload=st.file_uploader('Upload time and brightness columns',type=['csv'])
        if upload is None:
            st.info('CSV needs numeric time and flux/brightness columns. Time units are preserved; flux may be relative or instrumental.')
            return
        try:
            table=pd.read_csv(upload)
            if len(table.columns)<2:
                raise ValueError('The CSV needs at least two columns.')
            a,b=st.columns(2)
            with a: time_column=st.selectbox('Time column',list(table.columns))
            with b: flux_column=st.selectbox('Brightness column',list(table.columns),index=1)
            if time_column==flux_column:
                raise ValueError('Choose different time and brightness columns.')
            time=pd.to_numeric(table[time_column],errors='coerce').to_numpy(dtype=float)
            flux=pd.to_numeric(table[flux_column],errors='coerce').to_numpy(dtype=float)
            finite=np.isfinite(time)&np.isfinite(flux)
            if not finite.all():
                st.warning(f'Dropped {int((~finite).sum())} rows with missing or nonnumeric time/brightness.')
            time,flux=validate_curve(time[finite],flux[finite]); time_label=f'Time ({time_column}; supplied units)'
        except (ValueError,pd.errors.ParserError,pd.errors.EmptyDataError,UnicodeDecodeError) as exc:
            st.error(str(exc)); return
        st.caption('Uploaded observations: quality flags and uncertainties are not applied automatically.')
    elif selected.startswith('Observed'):
        table=pd.read_csv(observed); time,flux=validate_curve(table['time_bkjd'],table['pdcsap_flux'])
        time_label='Time (BKJD days)'
        st.caption('Observed Kepler-8, quarter 0: BKJD = BJD − 2454833 (TDB days); PDCSAP flux, finite samples with SAP_QUALITY = 0. Bundled quality-filtered series; no runtime download. [Source and preparation](https://registry.opendata.aws/mast-kepler/) · data/README.md records the exact product and selection.')
    else:
        time,flux=simulated_curve(selected)
        st.caption('Simulated educational signal with reproducible noise. This is not an observed star or a fitted physical model.')
    if len(time)>1024:
        start=st.slider('First sample to inspect',0,len(time)-128,0)
        time,flux=time[start:start+1024],flux[start:start+1024]
        st.caption('Displaying up to 1,024 consecutive samples without resampling.')
    operation=st.selectbox('Light-curve operation',['Convolution','Max pooling','Average pooling','Minimum pooling','Activation'])
    preprocessing=st.selectbox('Brightness scaling',['Relative flux (divide by median)','Original values','Center at median'])
    baseline=float(np.median(flux))
    if preprocessing.startswith('Relative'):
        if abs(baseline)<1e-12:
            st.error('Median brightness is zero; choose original values or centering.'); return
        flux=flux/baseline
    elif preprocessing=='Center at median':
        flux=flux-baseline
    kernel=None; activation='ReLU'; window=4; stride=1
    a,b=st.columns(2)
    if operation=='Convolution':
        with a: kernel_name=st.selectbox('1D kernel',list(KERNELS)); kernel=KERNELS[kernel_name]
        with b: stride=st.slider('Sample stride',1,16,1)
        st.caption('CNN-style cross-correlation: multiply samples by the kernel in its displayed order, then sum. No kernel flip or time interpolation.')
        st.dataframe(pd.DataFrame({'Kernel weight':kernel}),width='stretch')
    elif operation=='Activation':
        activation=st.selectbox('Light-curve activation',['ReLU','Leaky ReLU','Sigmoid','Tanh'])
        if preprocessing!='Center at median' and activation=='ReLU':
            st.info('Positive flux passes through ReLU unchanged. Center at the median to inspect how negative deviations are clipped.')
    else:
        with a: window=st.slider('Samples per window',2,min(32,len(time)),4 if len(time)>=4 else 2)
        with b: stride=st.slider('Sample stride',1,32,window)
        st.info('Pooling can alter transit depth: max pooling may remove dips; minimum pooling preserves low samples but also low outliers. These are teaching operations, not a transit-detection pipeline.')
    skip_gaps=st.checkbox('Skip windows crossing gaps larger than 3 × median cadence',value=True)
    if np.max(np.diff(time))>3*np.median(np.diff(time)):
        st.warning('There are time gaps. Windows use sample order, not a fixed time duration. No interpolation is performed.')
    try:
        result=operate_curve(time,flux,operation,window,stride,kernel,activation,skip_gaps)
    except ValueError as exc:
        st.error(str(exc)); return
    st.caption(f"{len(time)} input samples → {len(result['output'])} output values · {result['skipped']} gap-crossing windows skipped · output time is the mean time within each window.")
    figure=curve_animation(result,operation,time_label)
    st.plotly_chart(figure,width='stretch',key='light_curve_animation',config={'displayModeBar':False})
    st.caption('Gold marks the active window; mint diamonds mark selected maxima/minima. Playback shows up to 80 checkpoints; every output value is computed. Incomplete trailing windows are skipped.')
    with st.expander('Inspect numerical output'):
        output=pd.DataFrame({'time':result['output_time'],'value':result['output']})
        st.dataframe(output,width='stretch')
        st.download_button('Download processed values',output.to_csv(index=False).encode(),file_name='light-curve-output.csv',mime='text/csv')
    st.download_button('Download light-curve animation',figure.to_html(include_plotlyjs=True).encode(),file_name='light-curve-animation.html',mime='text/html')
