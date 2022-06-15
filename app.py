import streamlit as st
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import plotly.express as px


st.title("Bu model exokardiogramma tasvirlarini klassifikatsiya qiluvchi model")

file = st.file_uploader('Rasm yuklash', type=['png', 'jpg', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)

    img = PILImage.create(file)

    model = load_learner('Echocardiography_model.pkl')


    pred, pred_id, probs =  model.predict(img)
    foiz = probs[pred_id]*100
    if foiz>80:
        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

        fig = px.bar(x=probs*100, y=model.dls.vocab)
        st.plotly_chart(fig)
    else:
        st.markdown("Bu rasm exokardiogramma tasviri emas")

