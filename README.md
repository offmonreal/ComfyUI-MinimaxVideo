# ComfyUI-MinimaxVideo

A third-party custom node plugin for ComfyUI that provides seamless integration with the latest video generation models on the **MiniMax (Hailuo)** open platform (Hailuo-2.3, Hailuo-02, T2V-01, S2V-01, etc.), enabling fast creation of high-quality AI videos.

## ✨ Key Features

- **Three mainstream generation modes**:
  - **📝 Text-to-Video**: Supports 15 professional camera-move command syntax tokens such as `[pan_left,tilt_up]` for cinematic camera control.
  - **🖼️ Image-to-Video**: Pass a single reference image / first-frame image to turn a still picture into a dynamic clip.
  - **🧷 First-and-Last-Frame Video**: Provide both a first frame and a last frame to precisely control the start and end of the video (currently supported exclusively by the `MiniMax-Hailuo-02` model; the node automatically switches to that model to guarantee generation).
- **Full alignment with the latest official API parameters**: Supports duration (6s / 10s), resolution (768P / 1080P / 720P), automatic prompt optimization toggle, watermark toggle, and other advanced options.
- **Privacy-safe (zero persistence)**: Never commit your API key to the codebase. Load the key flexibly via the node UI, the `MINIMAX_API_KEY` environment variable, or a local `config.json`.
- **Fully asynchronous polling and download**: The backend uses an asynchronous polling mechanism to fetch task status. On success, it automatically calls the File API to download and save the result to the ComfyUI output directory.

## 📦 Installation

1. Enter your ComfyUI `custom_nodes` directory.
2. Clone this repository:
   ```bash
   git clone https://github.com/rickSF/Comfyui-MinimaxVideo.git
   ```
   *(Or click `Code -> Download ZIP` in the upper-right corner, then unzip the folder into the `custom_nodes` directory.)*
3. **Restart ComfyUI**.

## ⚙️ Configuring the API Key

There are three ways to configure the API key for the node. The priority from highest to lowest is:

1. **Enter directly on the node**: In the ComfyUI interface, type the key into the `🔑API Key` input box (highest priority, best for quick testing).
2. **Environment variable**: Add the `MINIMAX_API_KEY` environment variable to your system.
3. **Local config file** (recommended for long-term use):
   In the plugin root directory, copy `config.json.example` and rename it to `config.json`, then fill in your API key:
   ```json
   {
       "MINIMAX_API_KEY": "YOUR_REAL_API_KEY"
   }
   ```
   *Note: `config.json` is included in `.gitignore`, so it will never be accidentally committed to GitHub.*

## 💡 Usage Guide

1. Launch ComfyUI.
2. Right-click on an empty area / double-click to search, then locate and add the node under category `🤖Dapao-Toolbox` (or your custom category): **`🤖Minimax Video Generate`**.
3. Choose the `🎬Generation Mode` you need:
   - **Text-to-Video**: Fill in the `📝Prompt`.
   - **Image-to-Video**: Connect a native `Load Image` node to the `🖼️First Frame Image` or `🖼️Reference Image (for Image-to-Video)` input of this node.
   - **First-and-Last-Frame Video**: Both `🖼️First Frame Image` and `🖼️Last Frame Image` must be connected.

## ⚠️ Notes

- **First-and-Last-Frame model limitation**: Officially, only the `MiniMax-Hailuo-02` model supports first-and-last-frame generation. If you select the first-and-last-frame mode without picking that model, this node will automatically switch to it in the background and print a notice to the console.
- **Troubleshooting tips**: If your workflow contains other third-party image-processing nodes (such as upload nodes from RunningHub and similar services), please remove them if possible and use the native ComfyUI `Load Image` node connected to this node directly. This avoids authentication or network errors from third-party nodes interfering with generation.

## 📄 License

MIT License
