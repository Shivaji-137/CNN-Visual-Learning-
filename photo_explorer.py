"""Photograph examples and full-image operation animations, without downloads."""
import numpy as np
from PIL import Image


def sample_photos():
    from sklearn.datasets import load_sample_image
    photos = []
    for title, filename in [('Garden photograph', 'china.jpg'), ('Flower photograph', 'flower.jpg')]:
        image = Image.fromarray(load_sample_image(filename))
        image.thumbnail((128, 128), Image.Resampling.LANCZOS)
        owner, photo_id = ('danielbuechele', '6061409035') if filename == 'china.jpg' else ('vultilion', '6056698931')
        photo_url = f'https://www.flickr.com/photos/{owner}/{photo_id}/'
        image.info['credit'] = f'Photo: [{owner}]({photo_url}) · [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) · resized; processing views use grayscale.'
        image.info['credit_html'] = f'Photo: <a href="{photo_url}">{owner}</a> · <a href="https://creativecommons.org/licenses/by/2.0/">CC BY 2.0</a> · resized and processed in grayscale.'
        photos.append(dict(name=title, image=image))
    return photos


def photo_steps(source, operation, kernel=None, window=2, activation='ReLU', checkpoints=48):
    """Compute every output value; retain bounded animation checkpoints only."""
    from operations_explorer import activation_values
    source = np.asarray(source, dtype=float)
    if source.ndim != 2 or not source.size:
        raise ValueError('Expected a nonempty grayscale image or feature map.')
    stride = window if operation in {'Max pooling', 'Average pooling'} else 1
    if operation in {'Convolution', 'Max pooling', 'Average pooling'}:
        size = np.asarray(kernel).shape[0] if operation == 'Convolution' else window
        if size < 1 or min(source.shape) < size:
            raise ValueError('The operation window must fit the image.')
        regions = np.lib.stride_tricks.sliding_window_view(source, (size, size))[::stride, ::stride]
        if operation == 'Max pooling':
            output = regions.max(axis=(-2, -1))
        elif operation == 'Average pooling':
            output = regions.mean(axis=(-2, -1))
        else:
            output = np.einsum('ijab,ab->ij', regions, np.asarray(kernel))
    elif operation == 'Activation':
        size = 1
        output = activation_values(source, activation)
    else:
        raise ValueError('Choose convolution, activation, or pooling.')
    positions = np.unique(np.linspace(0, output.size-1, min(checkpoints, output.size), dtype=int))
    steps = []
    for position in positions:
        r, c = np.unravel_index(position, output.shape)
        revealed = np.full(output.shape, np.nan)
        revealed.flat[:position+1] = output.flat[:position+1]
        top, left = int(r*stride), int(c*stride)
        region = source[top:top+size, left:left+size]
        value = output[r, c]
        formula = {'Max pooling': 'max(window)', 'Average pooling': 'mean(window)',
                   'Convolution': 'sum(window × kernel)', 'Activation': f'{activation}(pixel)'}[operation]
        steps.append(dict(top=top, left=left, size=size, row=int(r), col=int(c),
                          output=revealed, position=int(position), detail=f'{formula} = {value:.3f}'))
    return output, steps


def photo_animation(source, output, steps, operation, photograph, source_stride=1, receptive_field=1):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Original photograph', 'Operation input', 'Image being built'), horizontal_spacing=.05)
    photograph = np.asarray(photograph)
    if photograph.ndim == 2:
        photograph = np.repeat(photograph[:, :, None], 3, axis=2)
    if photograph.dtype != np.uint8:
        photograph = (np.clip(photograph, 0, 1)*255).astype(np.uint8)
    fig.add_trace(go.Image(z=photograph), 1, 1)
    low, high = float(min(source.min(), output.min())), float(max(source.max(), output.max()))
    if high == low:
        high = low+1
    colors = 'RdBu' if low < 0 else 'Gray'
    for column, data in [(2, source), (3, steps[0]['output'])]:
        fig.add_trace(go.Heatmap(z=data, colorscale=colors, zmin=low, zmax=high, showscale=False,
                                hovertemplate='row %{y}, column %{x}: %{z:.3f}<extra></extra>', hoverongaps=False), 1, column)

    def outlines(step):
        top, left, size = step['top'], step['left'], step['size']
        return [dict(type='rect', xref='x', yref='y', x0=left*source_stride-.5, y0=top*source_stride-.5,
                     x1=(left+size-1)*source_stride+receptive_field-.5,
                     y1=(top+size-1)*source_stride+receptive_field-.5, line=dict(color='#fbbf24',width=3)),
                dict(type='rect', xref='x2', yref='y2', x0=left-.5, y0=top-.5,
                     x1=left+size-.5,y1=top+size-.5,line=dict(color='#fbbf24',width=3)),
                dict(type='rect', xref='x3', yref='y3', x0=step['col']-.5,y0=step['row']-.5,
                     x1=step['col']+.5,y1=step['row']+.5,line=dict(color='#fbbf24',width=3))]

    def title(step):
        return dict(text=f"{operation} · {step['position']+1:,}/{output.size:,} values<br><sup>{step['detail']}</sup>",x=.5,xanchor='center',font=dict(size=17))

    fig.frames = [go.Frame(name=str(i), traces=[2], data=[go.Heatmap(z=step['output'])],
                          layout=go.Layout(shapes=outlines(step),title=title(step))) for i,step in enumerate(steps)]
    instant = dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))
    fig.update_layout(template='plotly_dark', height=520, paper_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=10,r=10,t=100,b=120),title=title(steps[0]),shapes=outlines(steps[0]),
                      updatemenus=[dict(type='buttons',direction='left',x=0,y=-.12,showactive=False,bgcolor='#142033',
                        buttons=[dict(label='▶ Play',method='animate',args=[None,dict(fromcurrent=True,frame=dict(duration=150,redraw=True),transition=dict(duration=0))]),
                                 dict(label='Ⅱ Pause',method='animate',args=[[None],instant]),
                                 dict(label='↺ Restart',method='animate',args=[['0'],instant])])],
                      sliders=[dict(x=.03,len=.94,y=-.25,currentvalue=dict(prefix='Checkpoint '),
                        steps=[dict(label=str(i+1),method='animate',args=[[str(i)],instant]) for i in range(len(steps))])])
    for col,data in [(1,photograph),(2,source),(3,output)]:
        xaxis = 'x' if col == 1 else f'x{col}'
        fig.update_xaxes(showticklabels=False,showgrid=False,range=[-.5,data.shape[1]-.5],fixedrange=True,constrain='domain',row=1,col=col)
        fig.update_yaxes(showticklabels=False,showgrid=False,range=[data.shape[0]-.5,-.5],autorange=False,
                         fixedrange=True,scaleanchor=xaxis,scaleratio=1,constrain='domain',row=1,col=col)
    return fig


def render_photo_operation(source, photograph, operation, kernel=None, window=2, activation='ReLU', key='photo', receptive_field=1):
    import streamlit as st
    factor = max(1, int(np.ceil(max(source.shape)/128)))
    if factor > 1:
        source = source[::factor, ::factor]
        st.caption(f"Large feature map: this animation samples every {factor} pixels before applying the operation. The full-resolution processing remains above.")
    output,steps=photo_steps(source,operation,kernel,window,activation)
    figure=photo_animation(source,output,steps,operation,photograph,source_stride=factor,receptive_field=receptive_field)
    st.plotly_chart(figure,width='stretch',config={'displayModeBar':False},key=f'{key}_photo_animation')
    st.caption(f"{source.shape[1]} × {source.shape[0]} input → {output.shape[1]} × {output.shape[0]} output. Every value is computed; {len(steps)} playback checkpoints keep the animation responsive. Gold boxes link the photograph, active window, and output. Shared brightness scale preserves differences between input and output.")
    credit = getattr(photograph, 'info', {}).get('credit')
    html = figure.to_html(include_plotlyjs=True)
    if credit:
        st.caption(credit)
        credit_html = photograph.info['credit_html']
        html = html.replace('<body>', f'<body><p>{credit_html}</p>', 1)
    st.download_button('Download photo animation',html.encode(),
                       file_name='photo-animation.html',mime='text/html',key=f'{key}_photo_download')


def render():
    import streamlit as st
    from scipy.signal import correlate2d
    st.title('Photo Lab')
    st.write('Watch a real photograph change as a window moves across it. Pooling reduces detail, convolution finds edges or blurs, and activation changes the responses.')
    photos=sample_photos()
    selected=st.selectbox('Photograph',[photo['name'] for photo in photos]+['Upload a photograph'])
    if selected == 'Upload a photograph':
        upload=st.file_uploader('Choose a photograph',type=['jpg','jpeg','png'])
        if upload is None:
            return
        from PIL import ImageOps
        try:
            photograph=ImageOps.exif_transpose(Image.open(upload)).convert('RGB')
            photograph.thumbnail((128,128),Image.Resampling.LANCZOS)
        except (OSError,ValueError):
            st.error('This image could not be opened. Try another JPEG or PNG.')
            return
    else:
        photograph=next(photo['image'] for photo in photos if photo['name']==selected)
    gray=np.asarray(photograph.convert('L'),dtype=float)/255
    operation=st.selectbox('Photo operation',['Max pooling','Average pooling','Convolution','Activation'])
    kernels={'Find vertical edges':np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),
             'Find horizontal edges':np.array([[-1,-1,-1],[0,0,0],[1,1,1]]),
             'Blur':np.ones((3,3))/9,'Sharpen':np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])}
    kernel=None; window=2; activation='ReLU'; source=gray; receptive=1
    if operation in {'Max pooling','Average pooling'}:
        window=st.slider('Pooling window',2,8,4)
        if min(gray.shape)<window:
            st.info('Choose a photograph larger than the pooling window.')
            return
        st.info('The smaller output is enlarged for comparison. Watch fine textures merge while stronger light regions remain.')
    else:
        if min(gray.shape)<3:
            st.info('Choose a photograph at least 3 pixels wide and high.')
            return
        kernel=kernels[st.selectbox('Image filter',list(kernels))]
        if operation=='Activation':
            source=correlate2d(gray,kernel,mode='valid')
            receptive=3
            activation=st.selectbox('Photo activation',['ReLU','Leaky ReLU','Sigmoid','Tanh'])
            st.info('Activation operates on the filtered image. ReLU removes negative responses; compare the feature map with the original photo.')
    render_photo_operation(source,photograph,operation,kernel,window,activation,key='photo_lab',receptive_field=receptive)
    st.caption('Photos are reduced to at most 128 pixels per side for playback, then processed in grayscale. Built-in garden and flower photos ship with scikit-learn; no download is needed.')
