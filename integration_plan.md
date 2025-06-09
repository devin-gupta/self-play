# Plan to Integrate Diffusion Model into CustomFrankaEnv

This document outlines the steps to integrate the `Inference` class from `src/diffusion/inference.py` into the `CustomFrankaEnv` class in `src/devin_experiments/franka/utils/custom_franka_env.py`. The goal is to dynamically generate a goal image using a diffusion model instead of loading a static one.

## Step 1: Modify `CustomFrankaEnv` to Initialize the Inference Model

The `CustomFrankaEnv` class needs to be aware of the diffusion model. We will load the model during the environment's initialization.

**File to modify:** `src/devin_experiments/franka/utils/custom_franka_env.py`

**Actions:**

1.  **Import the `Inference` class.** Add the following import statement at the top of the file. You may need to add `sys.path` modifications if the import fails.
    ```python
    from src.diffusion.inference import Inference
    ```

2.  **Update the `__init__` method.**
    *   Add a `model_checkpoint_path` parameter to the constructor.
    *   Instantiate the `Inference` class and store it as an instance attribute `self.inference`.

    **Current `__init__`:**
    ```python
    def __init__(self, env):
        super().__init__(env)
        self.base_env = env
        # ... rest of the method
    ```

    **New `__init__`:**
    ```python
    def __init__(self, env, model_checkpoint_path: str):
        super().__init__(env)
        self.base_env = env
        # ... rest of the observation space definition ...
        self.action_space = self.base_env.action_space
        
        # Initialize the inference model
        self.inference = Inference(model_checkpoint_path)
    ```

## Step 2: Update `get_goal_image` to Use the Diffusion Model

Instead of loading a static goal image from a file, we will now use the loaded diffusion model to predict a goal image based on the current state.

**File to modify:** `src/devin_experiments/franka/utils/custom_franka_env.py`

**Actions:**

1.  **Rewrite the `get_goal_image` method.**
    *   Get the current image from `self.get_curr_image()`.
    *   The `predict` method of the `Inference` class expects a `(1, 120, 120)` NumPy array of type `np.uint8`. The `get_curr_image` method returns a `(120, 120)` array. We need to add a channel dimension.
    *   Call `self.inference.predict()` with the correctly shaped current image.
    *   The `predict` method returns a `(1, 120, 120)` array. The rest of the `CustomFrankaEnv` expects a `(120, 120)` array for the goal image. We need to squeeze the channel dimension from the output.

    **Current `get_goal_image`:**
    ```python
    def get_goal_image(self):
        img = Image.open('goal_image.png')
        img = torchvision.transforms.functional.rgb_to_grayscale(img, num_output_channels=1)
        img = torchvision.transforms.functional.resize(img, (120, 120))
        return np.array(img, dtype = np.uint8)
    ```

    **New `get_goal_image`:**
    ```python
    def get_goal_image(self):
        # Get the current state image
        current_image = self.get_curr_image() # Shape: (120, 120), dtype: uint8

        # Add a channel dimension to match the model's expected input shape (1, 120, 120)
        current_image_expanded = np.expand_dims(current_image, axis=0)
        
        # Predict the goal image using the diffusion model
        # The predict method returns a (1, 120, 120) numpy array of dtype uint8
        predicted_goal_image = self.inference.predict(current_image_expanded)

        # Remove the channel dimension to get shape (120, 120) as used in the rest of the class
        goal_img = predicted_goal_image.squeeze(0)
        
        return goal_img
    ```

## Step 3: Find and Update `CustomFrankaEnv` Instantiation

Somewhere in the codebase, `CustomFrankaEnv` is instantiated. We need to find this location and pass the `model_checkpoint_path` to the constructor.

**Action:**

1.  **Search the codebase** for occurrences of `CustomFrankaEnv(`.
2.  **Update the instantiation call.** You will need to provide a valid path to a model checkpoint file (`.pth`).

    **Example:**

    If you find this code:
    ```python
    env = CustomFrankaEnv(base_env)
    ```

    You need to change it to something like this:
    ```python
    MODEL_PATH = "path/to/your/model.pth" # This path needs to be determined
    env = CustomFrankaEnv(base_env, model_checkpoint_path=MODEL_PATH)
    ```

This concludes the plan. Following these steps will successfully integrate the diffusion model for dynamic goal generation within the Franka environment wrapper. 