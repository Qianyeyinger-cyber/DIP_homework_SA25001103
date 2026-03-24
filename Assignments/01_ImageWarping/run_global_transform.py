import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])


# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    if image is None:
        return None

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)

    # Ensure 3 channels
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros(
        (pad_size * 2 + image.shape[0], pad_size * 2 + image.shape[1], 3),
        dtype=np.uint8
    ) + np.array((255, 255, 255), dtype=np.uint8).reshape(1, 1, 3)

    image_new[
        pad_size:pad_size + image.shape[0],
        pad_size:pad_size + image.shape[1]
    ] = image

    image = np.array(image_new)
    h, w = image.shape[:2]

    # Image center
    cx, cy = w / 2.0, h / 2.0

    ### Apply Composition Transform
    # 1) rotation + scale around image center
    M_rs_2x3 = cv2.getRotationMatrix2D((cx, cy), rotation, scale)
    M_rs = to_3x3(M_rs_2x3)

    # 2) horizontal flip around image center
    if flip_horizontal:
        # x' = -x + (w - 1), y' = y
        M_flip = np.array([
            [-1,  0, w - 1],
            [ 0,  1, 0],
            [ 0,  0, 1]
        ], dtype=np.float32)
    else:
        M_flip = np.eye(3, dtype=np.float32)

    # 3) translation
    M_trans = np.array([
        [1, 0, translation_x],
        [0, 1, translation_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # Final composition:
    # first rotate/scale, then flip, then translate
    M = M_trans @ M_flip @ M_rs

    # Convert back to 2x3 for warpAffine
    M_affine = M[:2, :]

    transformed_image = cv2.warpAffine(
        image,
        M_affine,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    return transformed_image


# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")

        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")

            # Right: Output image
            image_output = gr.Image(label="Transformed Image")

        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation,
            translation_x, translation_y,
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo


# Launch the Gradio interface
interactive_transform().launch()
