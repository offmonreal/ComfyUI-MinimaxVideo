# ComfyUI-MinimaxVideo

A ComfyUI custom node plugin that provides two nodes for the MiniMax video
generation platform:

- **MiniMax Hailuo Video (V1)** — the V1 `video_generation` API.
- **MiniMax H3 Video (V2)** — the V2 `video_generation` API.

Both nodes live in the same package and are discovered through a single
`__init__.py`, so installing this repository under `ComfyUI/custom_nodes`
exposes both nodes at once.

---

## 📦 Installation

1. Enter your ComfyUI `custom_nodes` directory.
2. Clone this repository:
   ```bash
   git clone https://github.com/offmonreal/ComfyUI-MinimaxVideo.git
   ```
3. Restart ComfyUI.

No extra Python dependencies are required beyond what ComfyUI itself ships
(`numpy`, `Pillow`, `torch`).

---

## 🔑 API Key

Each node resolves its API key in this order:

1. The value typed into the node's API-key widget.
2. The `MINIMAX_API_KEY` environment variable.
3. The `MINIMAX_API_KEY` field in `config.json` next to `__init__.py`.

Both Token Plan subscription keys (`sk-cp-...`) and normal PAYG keys are
accepted by the V1 node. The V2 node is PAYG only. The key is forwarded
verbatim to the server with `Authorization: Bearer <key>`; the server
decides which type of key it is.

Example `config.json`:
```json
{
    "MINIMAX_API_KEY": "YOUR_REAL_API_KEY"
}
```

`config.json` is listed in `.gitignore`, so it will never be committed.

---

## 🎬 MiniMax Hailuo Video (V1) — `MinimaxVideoGenerate`

Targets the V1 endpoint family:

- Create: `POST /v1/video_generation`
- Poll:   `GET  /v1/query/video_generation?task_id=...`
- File:   `GET  /v1/files/retrieve?file_id=...`

### Models

- `MiniMax-Hailuo-2.3`
- `MiniMax-Hailuo-2.3-Fast`
- `MiniMax-Hailuo-02`

### Modes

- **📝 Text-to-Video** — prompt is required.
- **🖼️ Image-to-Video** — prompt is optional; a `Reference Image` or
  `First Frame Image` is required.
- **🧷 First + Last Frame** — prompt is optional; both a `First Frame
  Image` and a `Last Frame Image` are required. Only `MiniMax-Hailuo-02`
  supports this mode in V1.

### Resolutions

Choices on the node: `512P`, `768P`, `1080P`. Each model accepts a subset of
these. Concretely:

| Model                    | Modes            | Resolutions       | Durations           |
|--------------------------|------------------|-------------------|---------------------|
| `Hailuo-2.3`             | T2V              | 768P, 1080P       | 6s, 10s / 6s        |
| `Hailuo-2.3`             | I2V              | 768P, 1080P       | 6s, 10s / 6s        |
| `Hailuo-2.3-Fast`        | I2V only         | 768P, 1080P       | 6s, 10s / 6s        |
| `Hailuo-02`              | T2V              | 768P, 1080P       | 6s, 10s / 6s        |
| `Hailuo-02`              | I2V              | 512P, 768P, 1080P | 6s, 10s / 6s, 10s / 6s |
| `Hailuo-02`              | First + Last     | 768P, 1080P       | 6s, 10s / 6s        |

512P is explicitly disallowed for First + Last Frame in V1.

### Other V1 options

- `✨ Prompt Optimizer` — sent as `prompt_optimizer`.
- `⚡ Fast Preprocessing` — sent as `fast_pretreatment` for T2V and I2V.
  Not sent for First + Last Frame (not documented there).
- `🌐 Base URL` — defaults to `https://api.minimax.io`.
- `⌛ Max Wait (seconds)` and `🔁 Poll Interval (seconds)` — control the
  V1 polling loop.

### V1 success flow

1. POST to `/v1/video_generation`.
2. Poll `/v1/query/video_generation?task_id=...` until `status == "Success"`.
3. Read `file_id` from the query response.
4. Call `/v1/files/retrieve?file_id=...` to obtain `download_url`.
5. Download the MP4 into the ComfyUI output directory.

---

## 🎞️ MiniMax H3 Video (V2) — `MinimaxH3VideoGenerate`

Targets the V2 endpoint family:

- Create: `POST /v2/video_generation`
- Poll:   `GET  /v2/query/video_generation/{task_id}`

V2 is PAYG only.

### Model

- `MiniMax-H3` (single fixed model; no dropdown).

### Modes

- **📝 Text-to-Video** — prompt required; aspect ratio must be explicit.
  `adaptive` is rejected.
- **🖼️ First Frame Image-to-Video** — `adaptive` only; `First Frame Image`
  required.
- **🖼️ Last Frame Image-to-Video** — `adaptive` only; `Last Frame Image`
  required.
- **🧷 First + Last Frame** — `adaptive` only; both frame images required.
- **🖼️ Reference Image-to-Video** — `adaptive` or any explicit supported
  ratio; `Reference Image` required.

### Resolutions

`768P`, `2K`.

### Durations

Integer seconds, **4 through 15 inclusive**.

### Aspect ratios

`adaptive`, `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`.

The widget label says "Aspect Ratio"; the wire field is `ratio`
(the official MiniMax V2 parameter name).

### V2 request shape

The node sends a `content` array — one `text` item plus zero or more
`image_url` items with `role` set to `first_frame`, `last_frame`, or
`reference_image`. On success, the video URL is read from
`task.content.url` of the polling response.

---

## 🔐 V1 vs V2

| Aspect            | V1 (`MinimaxVideoGenerate`)        | V2 (`MinimaxH3VideoGenerate`)          |
|-------------------|------------------------------------|----------------------------------------|
| Endpoints         | `/v1/video_generation`, `/v1/query/video_generation`, `/v1/files/retrieve` | `/v2/video_generation`, `/v2/query/video_generation/{task_id}` |
| Models            | Hailuo-2.3 / 2.3-Fast / 02         | H3                                     |
| Resolutions       | 512P / 768P / 1080P                | 768P / 2K                              |
| Durations         | 6s, 10s                            | 4–15s                                  |
| Aspect ratio      | Not exposed                        | `ratio` parameter, `adaptive` allowed for image-conditioned modes |
| Request shape     | Flat fields + base64 frame images  | `content` array with `text` and `image_url` items |
| Success URL       | `file_id` -> `/v1/files/retrieve`  | `task.content.url` in the query response |
| Auth              | PAYG or Token Plan (`sk-cp-...`)   | PAYG only                              |

---

## 📄 License

MIT License. See `LICENSE`.
