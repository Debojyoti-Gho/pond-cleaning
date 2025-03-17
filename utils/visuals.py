import streamlit as st
import matplotlib.pyplot as plt

def display_segments(segments, base_image):
    fig, ax = plt.subplots()
    ax.imshow(base_image)
    for seg in segments:
        x, y, w, h = seg["bbox"]
        label = seg["label"]
        ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='lime', facecolor='none', lw=2))
        ax.text(x, y - 5, label, color='white', bbox=dict(facecolor='green', alpha=0.5))
    st.pyplot(fig)
