import streamlit as st
import numpy as np
from PIL import Image
import tempfile
from transformers import AutoImageProcessor, ViTModel
import torch

class Transforms:
    @staticmethod
    def rgb_to_lms():
        return np.array([[17.8824, 43.5161, 4.11935],
                         [3.45565, 27.1554, 3.86714],
                         [0.0299566, 0.184309, 1.46709]]).T

    @staticmethod
    def lms_to_rgb():
        return np.array([[0.0809, -0.1305, 0.1167],
                         [-0.0102, 0.0540, -0.1136],
                         [-0.0004, -0.0041, 0.6935]]).T

    @staticmethod
    def lms_protanopia_sim(degree: float = 1.0):
        return np.array([[1 - degree, 2.02344 * degree, -2.52581 * degree],
                         [0, 1, 0],
                         [0, 0, 1]]).T

    @staticmethod
    def lms_deutranopia_sim(degree: float = 1.0):
        return np.array([[1, 0, 0],
                         [0.494207 * degree, 1 - degree, 1.24827 * degree],
                         [0, 0, 1]]).T

    @staticmethod
    def lms_tritanopia_sim(degree: float = 1.0):
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [-0.395913 * degree, 0.801109 * degree, 1 - degree]]).T
    
    @staticmethod
    def correction_matrix(protanopia_degree, deutranopia_degree) -> np.ndarray:
        """
        Matrix for Correcting Colorblindness (protanomaly + deuteranomaly) from LMS color-space.
        :param protanopia_degree: Protanomaly degree for correction. If 0, correction is made for Deuteranomally only.
        :param deutranopia_degree: Deuteranomaly degree for correction. If 0, correction is made for Protanomaly only.
        """
        return np.array([[1 - deutranopia_degree/2, deutranopia_degree/2, 0],
                         [protanopia_degree/2, 1 - protanopia_degree/2, 0],
                         [protanopia_degree/4, deutranopia_degree/4, 1 - (protanopia_degree + deutranopia_degree)/4]]).T
    @staticmethod
    def hybrid_protanomaly_deuteranomaly_sim(degree_p: float = 1.0, degree_d: float = 1.0) -> np.ndarray:
        """
        Matrix for Simulating Hybrid Colorblindness (protanomaly + deuteranomaly) from LMS color-space.
        :param degree_p: protanomaly degree.
        :param degree_d: deuteranomaly degree.
        """
        return np.array([[1 - degree_p, 2.02344 * degree_p, -2.52581 * degree_p],
                         [0.494207 * degree_d, 1 - degree_d, 1.24827 * degree_d],
                         [0, 0, 1]]).T

class Utils:
    @staticmethod
    def load_rgb(path):
        img_rgb = np.array(Image.open(path)) / 255
        return img_rgb

    @staticmethod
    def load_lms(path):
        img_rgb = np.array(Image.open(path)) / 255
        img_lms = np.dot(img_rgb[:,:,:3], Transforms.rgb_to_lms())
        return img_lms


class ViTUtils:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')

    def extract_features(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0).numpy()


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
        img_features = Core.vit_utils.extract_features(img_rgb)  # ViT Feature extraction

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

    try:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            input_path = tempfile.NamedTemporaryFile(delete=False).name
            try:
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                image = Image.open(input_path)
                st.image(image, caption='Uploaded Image.', use_container_width=True)

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
                        try:
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

                        except Exception as e:
                            st.error(f"Error during image processing: {e}")
                            return

                    st.success('Processing completed!')

                    # Display the images and download options
                    display_and_download(simulated_img, 'Simulated Image', 'simulated_image.png')
                    if corrected_img is not None:
                        display_and_download(corrected_img, 'Corrected Image', 'corrected_image.png')

            except IOError:
                st.error("Error handling the file. Please try again with a valid image.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


def display_and_download(image_array, caption, file_name):
    st.image(image_array, caption=caption, use_container_width=True)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    Image.fromarray(image_array).save(tmp_file.name)
    st.download_button(label=f"Download {caption}",
                       data=open(tmp_file.name, 'rb').read(),
                       file_name=file_name,
                       mime='image/png')


if __name__ == '__main__':
    main()
