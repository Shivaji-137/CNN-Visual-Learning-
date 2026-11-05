"""Small, deterministic CNN operations with browser-side playback controls."""
import numpy as np


def window_steps(image, operation, window_size=2, stride=2, kernel=None):
    """Return valid-window output and snapshots; convolution uses CNN cross-correlation."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or window_size < 1 or stride < 1 or min(image.shape) < window_size:
        raise ValueError("Use a 2D image, a window that fits, and a positive stride.")
    if operation not in {"Max pooling", "Average pooling", "Convolution"}:
        raise ValueError("Unknown window operation.")
    if operation == "Convolution":
        kernel = np.asarray(kernel, dtype=float)
        if kernel.shape != (window_size, window_size):
            raise ValueError("Kernel dimensions must match the window size.")
    height, width = image.shape
    output = np.full(((height-window_size)//stride+1, (width-window_size)//stride+1), np.nan)
    steps = []
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            top, left = row*stride, col*stride
            region = image[top:top+window_size, left:left+window_size]
            if operation == "Max pooling":
                value = region.max()
                detail = f"max({', '.join(f'{x:g}' for x in region.flat)}) = {value:g}" if region.size <= 4 else f"max of {region.size} window values = {value:g}"
            elif operation == "Average pooling":
                value = region.mean()
                detail = f"sum {region.sum():g} ÷ {region.size} = {value:.2f}"
            else:
                value = np.sum(region*kernel)
                detail = f"sum(window × kernel) = {value:g}"
            output[row, col] = value
            steps.append(dict(top=top, left=left, size=window_size, row=row, col=col,
                              output=output.copy(), detail=detail))
    return output, steps


def activation_values(values, name):
    values = np.asarray(values, dtype=float)
    if name == "ReLU":
        return np.maximum(values, 0)
    if name == "Leaky ReLU":
        return np.maximum(.1*values, values)
    if name == "Sigmoid":
        # Stable for both large positive and negative inputs.
        return np.exp(-np.logaddexp(0, -values))
    if name == "Tanh":
        return np.tanh(values)
    raise ValueError("Unknown activation function.")


def point_steps(image, operation, activation="ReLU"):
    image = np.asarray(image, dtype=float)
    transformed = activation_values(image, activation) if operation == "Activation" else image.reshape(1, -1)
    revealed = np.full_like(transformed, np.nan)
    steps = []
    for index, value in enumerate(transformed.flat):
        row, col = np.unravel_index(index, image.shape)
        out_row, out_col = np.unravel_index(index, transformed.shape)
        revealed[out_row, out_col] = value
        detail = f"{activation}({image[row,col]:g}) = {value:.2f}" if operation == "Activation" else f"pixel [{row}, {col}] → vector [{index}] = {value:g}"
        steps.append(dict(top=row, left=col, size=1, row=out_row, col=out_col,
                          output=revealed.copy(), detail=detail))
    return transformed, steps


def animation_figure(image, output, steps, operation, speed=600):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Input", "Output"), horizontal_spacing=.13)
    input_limit = max(float(np.abs(image).max()), 1)
    output_limit = max(float(np.abs(output).max()), 1)
    fig.add_trace(go.Heatmap(z=image, zmin=-input_limit, zmax=input_limit, colorscale="RdBu", reversescale=True,
                            texttemplate="%{z:.2f}", showscale=False, hovertemplate="row %{y}, col %{x}: %{z:.2f}<extra></extra>"),1,1)
    fig.add_trace(go.Heatmap(z=steps[0]["output"], zmin=-output_limit, zmax=output_limit, colorscale="RdBu", reversescale=True,
                            texttemplate="%{z:.2f}", showscale=False, hoverongaps=False,
                            hovertemplate="row %{y}, col %{x}: %{z:.2f}<extra></extra>"),1,2)

    def outlines(step):
        return [dict(type="rect", xref="x", yref="y", x0=step['left']-.5, y0=step['top']-.5,
                     x1=step['left']+step['size']-.5, y1=step['top']+step['size']-.5,
                     line=dict(color="#e49b0f", width=5)),
                dict(type="rect", xref="x2", yref="y2", x0=step['col']-.5, y0=step['row']-.5,
                     x1=step['col']+.5, y1=step['row']+.5, line=dict(color="#e49b0f", width=5))]

    def heading(i, step):
        return dict(text=f"{operation} · Step {i+1}/{len(steps)}<br><sup>{step['detail']}</sup>", x=.5, xanchor="center", font=dict(size=16))

    fig.frames = [go.Frame(name=str(i), data=[go.Heatmap(z=step['output'])], traces=[1],
                          layout=go.Layout(shapes=outlines(step), title=heading(i,step))) for i,step in enumerate(steps)]
    play_args = dict(frame=dict(duration=speed, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")
    fig.update_layout(template="plotly_dark", font=dict(color="#e6edf5"), title=heading(0,steps[0]), shapes=outlines(steps[0]), height=500,
                      margin=dict(t=100,b=125,l=25,r=25), paper_bgcolor="rgba(0,0,0,0)",
                      updatemenus=[dict(type="buttons", direction="left", x=0, y=-.12, showactive=False, bgcolor="#142033", bordercolor="#456078", font=dict(color="#e6edf5"),
                        buttons=[dict(label="▶ Play", method="animate", args=[None,play_args]),
                                 dict(label="Ⅱ Pause", method="animate", args=[[None],dict(mode="immediate",frame=dict(duration=0,redraw=False))]),
                                 dict(label="↺ Restart", method="animate", args=[["0"],dict(mode="immediate",frame=dict(duration=0,redraw=True))])])],
                      sliders=[dict(active=0, x=.02, len=.96, y=-.26, currentvalue=dict(prefix="Step "),
                        steps=[dict(label=str(i+1),method="animate",args=[[str(i)],dict(mode="immediate",frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for i in range(len(steps))])])
    fig.update_xaxes(showgrid=False, tickmode="linear", dtick=1, side="top", constrain="domain")
    fig.update_yaxes(showgrid=False, autorange="reversed", tickmode="linear", dtick=1, constrain="domain")
    fig.update_xaxes(range=[-.5, image.shape[1]-.5], fixedrange=True, row=1,col=1)
    fig.update_yaxes(range=[image.shape[0]-.5, -.5], autorange=False, fixedrange=True, row=1,col=1)
    fig.update_xaxes(range=[-.5, output.shape[1]-.5], fixedrange=True,row=1,col=2)
    fig.update_yaxes(range=[output.shape[0]-.5, -.5], autorange=False, fixedrange=True,row=1,col=2)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1,col=1)
    if output.shape[0] > 1:
        fig.update_yaxes(scaleanchor="x2", scaleratio=1,row=1,col=2)
    return fig


def render():
    import streamlit as st
    st.title("Animation Lab")
    st.write("Play an operation, pause on any step, or move the timeline to inspect a calculation. Gold outlines connect the active input region to its output.")
    col1,col2,col3 = st.columns(3)
    with col1:
        operation = st.selectbox("Operation", ["Max pooling", "Average pooling", "Convolution", "Activation", "Flatten"])
    with col2:
        pattern = st.selectbox("Input pattern", ["Number grid", "Vertical edge", "Checkerboard", "Signed values"])
    with col3:
        size = st.slider("Image size",4,8,6)
    if pattern == "Number grid":
        image = np.arange(1,size*size+1,dtype=float).reshape(size,size) % 10
    elif pattern == "Vertical edge":
        image = np.zeros((size,size)); image[:,size//2:] = 8
    elif pattern == "Checkerboard":
        image = (np.indices((size,size)).sum(axis=0)%2)*8.0
    else:
        image = np.linspace(-5,5,size*size).reshape(size,size).round(1)
    activation = "ReLU"
    kernel = None
    with st.container(border=True):
        a,b,c = st.columns(3)
        if operation in {"Max pooling","Average pooling","Convolution"}:
            with a:
                window = st.slider("Window size",2,min(4,size),2) if operation != "Convolution" else 3
                if operation == "Convolution":
                    kernel_name = st.selectbox("Kernel",["Vertical edges","Horizontal edges","Sharpen","Box blur"])
                    kernel = {"Vertical edges": [[-1,0,1],[-1,0,1],[-1,0,1]],
                              "Horizontal edges": [[-1,-1,-1],[0,0,0],[1,1,1]],
                              "Sharpen": [[0,-1,0],[-1,5,-1],[0,-1,0]], "Box blur": np.ones((3,3))/9}[kernel_name]
            with b:
                stride = st.slider("Stride",1,3,1 if operation == "Convolution" else 2)
            output,steps = window_steps(image,operation,window,stride,kernel)
        else:
            with a:
                if operation == "Activation":
                    activation = st.selectbox("Activation",["ReLU","Sigmoid","Tanh"])
            output,steps = point_steps(image,operation,activation)
        with c:
            speed_label = st.select_slider("Playback speed",["Slow","Normal","Fast"],value="Normal")
    explanations = {
        "Max pooling": "Keep the largest value in each window. Strong responses survive while the spatial dimensions shrink.",
        "Average pooling": "Take the mean of each window. Every value contributes equally to the smoother output.",
        "Convolution": "Multiply each window by the kernel, then add the products. CNN libraries use this cross-correlation convention, without flipping the kernel.",
        "Activation": "Transform each value independently. ReLU clips negatives to zero; sigmoid maps to 0–1; tanh maps to −1–1.",
        "Flatten": "Read pixels row by row into a vector. Values stay the same; only the shape changes.",
    }
    st.info(explanations[operation])
    a,b,c = st.columns(3)
    a.metric("Input shape",f"{size} × {size}")
    b.metric("Output shape",f"{output.shape[0]} × {output.shape[1]}")
    c.metric("Steps",len(steps))
    if kernel is not None:
        with st.expander("Inspect kernel",expanded=True):
            st.dataframe(np.asarray(kernel),width="stretch")
    fig=animation_figure(image,output,steps,operation,{"Slow":1100,"Normal":600,"Fast":220}[speed_label])
    st.plotly_chart(fig,width="stretch",config={"displayModeBar":False},key="operation_animation")
    st.caption("Window operations use valid padding: border values that cannot form a complete window are skipped. The first calculation is shown before playback. Changing controls restarts the animation.")
    with st.expander("See the complete output"):
        st.dataframe(output,width="stretch")
    st.download_button("Download interactive animation",fig.to_html(include_plotlyjs=True).encode(),
                       file_name=f"cnn-{operation.lower().replace(' ','-')}.html",mime="text/html")


def render_feature_animation(image, kernel, convolution, activated, pooled,
                             activation_name, pooling_name, pool_size, key):
    """Animate a numeric crop of the real image pipeline without resampling values."""
    import streamlit as st
    with st.expander("Animate feature operations with numbers"):
        enabled = st.checkbox("Show numeric animation", key=f"{key}_enabled")
        if not enabled:
            st.caption("Explore the actual pixel and feature-map values with play, pause, and step controls.")
            return
        operations = ["Max pooling", "Average pooling", "Convolution", "Activation", "Flatten"]
        operation = st.selectbox("Feature operation", operations,
                                index=1 if pooling_name == "Average Pooling" else 0,
                                key=f"{key}_operation")
        source = {"Convolution":image, "Activation":convolution, "Flatten":pooled}.get(operation, activated)
        # A small crop keeps every number readable. No resizing or interpolation.
        crop_size = min(4 if operation == "Flatten" else 8, *source.shape)
        if operation == "Convolution" and min(source.shape) < max(kernel.shape):
            st.info("The image must be at least as large as the kernel.")
            return
        crop_size = max(crop_size, max(kernel.shape) if operation == "Convolution" else min(pool_size, *source.shape))
        strongest = np.unravel_index(np.abs(source).argmax(), source.shape)
        max_row, max_col = source.shape[0]-crop_size, source.shape[1]-crop_size
        start_row = max(0, min(max_row, int(strongest[0])-crop_size//2))
        start_col = max(0, min(max_col, int(strongest[1])-crop_size//2))
        col1,col2 = st.columns(2)
        with col1:
            row = st.slider("Crop top row",0,max_row,start_row,key=f"{key}_{operation}_row") if max_row else 0
        with col2:
            column = st.slider("Crop left column",0,max_col,start_col,key=f"{key}_{operation}_column") if max_col else 0
        crop = source[row:row+crop_size,column:column+crop_size]
        source_name = {"Convolution":"original grayscale image", "Activation":"convolution feature map", "Flatten":"pooled feature map"}.get(operation,f"feature map after {activation_name}")
        st.caption(f"Actual {crop_size} × {crop_size} crop of the {source_name}, starting at row {row}, column {column}. Values are not resized. Output below applies to this crop.")
        if operation in {"Max pooling","Average pooling"}:
            if min(crop.shape) < pool_size:
                st.info("Choose a smaller pooling size in the sidebar.")
                return
            output,steps=window_steps(crop,operation,pool_size,pool_size)
            st.caption(f"Window {pool_size} × {pool_size} · stride {pool_size} · valid padding")
        elif operation == "Convolution":
            output,steps=window_steps(crop,operation,kernel.shape[0],1,kernel)
            st.dataframe(np.asarray(kernel),width="stretch")
        elif operation == "Activation":
            output,steps=point_steps(crop,operation,activation_name)
        else:
            output,steps=point_steps(crop,operation)
        figure=animation_figure(crop,output,steps,operation)
        st.plotly_chart(figure,width="stretch",config={"displayModeBar":False},key=f"{key}_figure")
        st.download_button("Download this feature animation",figure.to_html(include_plotlyjs=True).encode(),
                           file_name=f"feature-{operation.lower().replace(' ','-')}.html",mime="text/html",key=f"{key}_download")
