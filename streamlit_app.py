import streamlit as st
import numpy as np
from PIL import Image
from utils import Transforms, Utils
from vit_utils import ViTUtils
import tempfile

class Core:
    vit_utils = ViTUtils()

    @staticmethod
    def simulate(input_path: str,
                 simulate_type: str = 'protanopia',
                 simulate_degree_primary: float = 1.0,
                 simulate_degree_sec: float = 1.0):
        assert simulate_type in ['protanopia', 'deutranopia', 'tritanopia', 'hybrid'], \
            'Invalid Simulate Type: {}'.format(simulate_type)

        img_rgb = Utils.load_rgb(input_path)
        img_features = Core.vit_utils.extract_features(img_rgb)
        
        if simulate_type == 'protanopia':
            transform = Transforms.lms_protanopia_sim(degree=simulate_degree_primary)
        elif simulate_type == 'deutranopia':
            transform = Transforms.lms_deutranopia_sim(degree=simulate_degree_primary)
        elif simulate_type == 'tritanopia':
            transform = Transforms.lms_tritanopia_sim(degree=simulate_degree_primary)
        else:
            transform = Transforms.hybrid_protanomaly_deuteranomaly_sim(degree_p=simulate_degree_primary,
                                                                        degree_d=simulate_degree_sec)

        img_lms = np.dot(img_rgb[:, :, :3], Transforms.rgb_to_lms())
        img_sim = np.dot(img_lms, transform)
        img_sim = np.uint8(np.dot(img_sim, Transforms.lms_to_rgb()) * 255)

        return img_sim

    @staticmethod
    def correct(input_path: str,
                protanopia_degree: float = 1.0,
                deutranopia_degree: float = 1.0):
        img_rgb = Utils.load_rgb(input_path)
        transform = Transforms.correction_matrix(protanopia_degree=protanopia_degree,
                                                 deutranopia_degree=deutranopia_degree)
        img_corrected = np.uint8(np.dot(img_rgb, transform) * 255)
        return img_corrected

def main():
    st.title("ReColorLib: Simulate and Correct Images for Color-Blindness")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        input_path = tempfile.NamedTemporaryFile(delete=False).name
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        image = Image.open(input_path)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        sim_type = st.selectbox(
            "Select Simulation Type:",
            ("protanopia", "deutranopia", "tritanopia", "hybrid")
        )
        
        simulate_degree_primary = st.slider(
            "Simulation Degree (Primary):", 0.0, 1.0, 1.0, 0.1)
        
        simulate_degree_sec = 1.0
        if sim_type == 'hybrid':
            simulate_degree_sec = st.slider(
                "Simulation Degree (Secondary):", 0.0, 1.0, 1.0, 0.1)
        
        correct_image = st.checkbox("Correct Image for Colorblindness")
        
        if st.button("Run"):
            with st.spinner('Processing...'):
                simulated_img = Core.simulate(
                    input_path=input_path,
                    simulate_type=sim_type,
                    simulate_degree_primary=simulate_degree_primary,
                    simulate_degree_sec=simulate_degree_sec
                )
                
                corrected_img = None
                if correct_image:
                    corrected_img = Core.correct(
                        input_path=input_path,
                        protanopia_degree=simulate_degree_primary,
                        deutranopia_degree=simulate_degree_sec
                    )
            
            st.success('Processing completed!')
            
            st.image(simulated_img, caption='Simulated Image.', use_column_width=True)
            
            tmp_sim = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            Image.fromarray(simulated_img).save(tmp_sim.name)
            st.download_button(label="Download Simulated Image",
                               data=open(tmp_sim.name, 'rb').read(),
                               file_name='simulated_image.png',
                               mime='image/png')
            
            if corrected_img is not None:
                st.image(corrected_img, caption='Corrected Image.', use_column_width=True)
                
                tmp_corr = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                Image.fromarray(corrected_img).save(tmp_corr.name)
                st.download_button(label="Download Corrected Image",
                                   data=open(tmp_corr.name, 'rb').read(),
                                   file_name='corrected_image.png',
                                   mime='image/png')
        
if __name__ == '__main__':
    main()
