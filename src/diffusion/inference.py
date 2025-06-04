import torch
import numpy as np
from typing import Any, Tuple
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt

class Inference:
    """
    A class for performing inference using a trained model checkpoint.

    The model is expected to take an image of the current state and return
    an image of the predicted next state.
    """

    MODEL_INPUT_SHAPE: Tuple[int, int, int, int] = (1, 1, 120, 120)  # (N, C, H, W)
    EXPECTED_IMAGE_SHAPE: Tuple[int, int, int] = (1, 120, 120) # (C, H, W)
    EXPECTED_IMAGE_DTYPE: np.dtype = np.uint8

    def __init__(self, model_checkpoint_path: str) -> None:
        """
        Initializes the Inference class by loading the model from a checkpoint.

        Args:
            model_checkpoint_path: Path to the .pth model checkpoint file.

        Raises:
            FileNotFoundError: If the model checkpoint file does not exist.
            RuntimeError: If there's an issue loading the model or it's not a valid type.
        """
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint file not found: {model_checkpoint_path}")

        try:
            # Determine device
            self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load the model. This could be a ScriptModule or a regular nn.Module.
            # For nn.Module, it's better if the model class is defined and then state_dict is loaded.
            # But for simplicity, we assume torch.load() loads the entire model.
            self.model: Any = torch.load(model_checkpoint_path, map_location=self.device)
            
            if not isinstance(self.model, (torch.nn.Module, torch.jit.ScriptModule)):
                raise RuntimeError(
                    f"Loaded object from {model_checkpoint_path} is not a "
                    f"torch.nn.Module or torch.jit.ScriptModule. Got type: {type(self.model)}"
                )
                
            self.model.to(self.device)
            self.model.eval() # Set the model to evaluation mode
            
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_checkpoint_path}: {e}") from e

    def predict(self, current_state_image: np.ndarray) -> np.ndarray:
        """
        Predicts the next state image based on the current state image.

        Args:
            current_state_image: A NumPy array representing the current state image.
                Shape: (1, 120, 120), dtype: np.uint8.

        Returns:
            A NumPy array representing the predicted next state image.
                Shape: (1, 120, 120), dtype: np.uint8.

        Raises:
            ValueError: If the input image does not match the expected shape or dtype.
            RuntimeError: If there's an issue during model inference.
        """
        # Input assertions
        if not isinstance(current_state_image, np.ndarray):
            raise ValueError(f"Input must be a NumPy array, got {type(current_state_image)}")
        if current_state_image.shape != self.EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                f"Input image shape must be {self.EXPECTED_IMAGE_SHAPE}, "
                f"got {current_state_image.shape}"
            )
        if current_state_image.dtype != self.EXPECTED_IMAGE_DTYPE:
            raise ValueError(
                f"Input image dtype must be {self.EXPECTED_IMAGE_DTYPE}, "
                f"got {current_state_image.dtype}"
            )

        # Preprocessing
        # 1. Convert to float and normalize to [0, 1]
        image_tensor: torch.Tensor = torch.from_numpy(current_state_image.astype(np.float32)) / 255.0
        # 2. Add batch dimension: (1, 120, 120) -> (1, 1, 120, 120)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        if image_tensor.shape != self.MODEL_INPUT_SHAPE:
             # This could happen if EXPECTED_IMAGE_SHAPE was (X, Y) instead of (C,X,Y)
             # Or if model expects different channel count.
            raise RuntimeError(
                f"Internal error: Processed tensor shape {image_tensor.shape} "
                f"does not match model input shape {self.MODEL_INPUT_SHAPE}."
            )

        # Inference
        try:
            with torch.no_grad(): # Disable gradient calculations
                predicted_tensor: torch.Tensor = self.model(image_tensor)
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {e}") from e

        # Postprocessing
        # 1. Ensure output tensor has 4 dimensions (N, C, H, W)
        if not (isinstance(predicted_tensor, torch.Tensor) and predicted_tensor.ndim == 4):
            raise RuntimeError(
                f"Model output is not a 4D tensor as expected. "
                f"Got shape {predicted_tensor.shape if isinstance(predicted_tensor, torch.Tensor) else type(predicted_tensor)}"
            )
        
        # 2. Remove batch dimension: (1, 1, 120, 120) -> (1, 120, 120)
        #    Also handle if model output is (N, C, H, W) where N > 1, C > 1
        #    For now, assume model output is (1,1,120,120) matching input channel/batch
        if predicted_tensor.shape[0] != 1 or predicted_tensor.shape[1] != self.EXPECTED_IMAGE_SHAPE[0]:
             raise RuntimeError(
                f"Model output tensor shape {predicted_tensor.shape} batch or channel size "
                f"does not match expected single output {self.MODEL_INPUT_SHAPE}. "
                f"Expected N=1, C={self.EXPECTED_IMAGE_SHAPE[0]}."
            )
        
        output_image_tensor: torch.Tensor = predicted_tensor.squeeze(0) # (1, 120, 120)

        # 3. Denormalize from [0, 1] to [0, 255] and convert to uint8
        output_image_np: np.ndarray = (output_image_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Output assertions
        if output_image_np.shape != self.EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                f"Output image shape must be {self.EXPECTED_IMAGE_SHAPE}, "
                f"got {output_image_np.shape}"
            )
        if output_image_np.dtype != self.EXPECTED_IMAGE_DTYPE:
            raise ValueError(
                f"Output image dtype must be {self.EXPECTED_IMAGE_DTYPE}, "
                f"got {output_image_np.dtype}"
            )

        return output_image_np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform inference using a trained model.")
    parser.add_argument("--model_checkpoint_path", type=str, help="Path to the .pth model checkpoint file.")
    parser.add_argument("--input_image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    # --- Load and preprocess the input image ---
    try:
        print(f"Loading input image from: {args.input_image_path}")
        img = Image.open(args.input_image_path).convert('L')  # Convert to grayscale (L mode)
        img = img.resize((120, 120), Image.Resampling.LANCZOS) # Resize (width, height)
        
        input_image_np = np.array(img, dtype=Inference.EXPECTED_IMAGE_DTYPE) # (120, 120)
        
        # Ensure shape is (1, 120, 120)
        if input_image_np.ndim == 2: # (H, W)
            input_image_np = np.expand_dims(input_image_np, axis=0) # (1, H, W)
        
        if input_image_np.shape != Inference.EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                f"Loaded image reshaped to {input_image_np.shape} but expected {Inference.EXPECTED_IMAGE_SHAPE}. "
                f"Original image path: {args.input_image_path}"
            )
        print(f"Input image loaded successfully. Shape: {input_image_np.shape}, dtype: {input_image_np.dtype}")

    except FileNotFoundError:
        print(f"Error: Input image file not found at {args.input_image_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading or processing input image: {e}")
        exit(1)

    print(f"\\nAttempting to initialize Inference class with model: {args.model_checkpoint_path}")
    try:
        inference_module = Inference(model_checkpoint_path=args.model_checkpoint_path)
        print("Inference class initialized successfully.")
        
        # Perform prediction
        print("Attempting prediction...")
        predicted_image_np = inference_module.predict(input_image_np)
        print(f"Predicted image shape: {predicted_image_np.shape}, dtype: {predicted_image_np.dtype}")
        print("Prediction successful.")

        # --- Display the images ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Input image display
        axes[0].imshow(input_image_np.squeeze(0), cmap='gray', vmin=0, vmax=255)
        axes[0].set_title(f"Input Image\\n{os.path.basename(args.input_image_path)}")
        axes[0].axis('off')
        
        # Predicted image display
        axes[1].imshow(predicted_image_np.squeeze(0), cmap='gray', vmin=0, vmax=255)
        axes[1].set_title("Predicted Next State")
        axes[1].axis('off')
        
        plt.suptitle(f"Model: {os.path.basename(args.model_checkpoint_path)}")
        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: Model checkpoint file not found. {e}")
    except RuntimeError as e:
        print(f"Runtime error during inference: {e}")
    except ValueError as e:
        print(f"Value error during inference: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 