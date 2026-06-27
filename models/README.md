# models/

The head tracker needs Google's **MediaPipe FaceLandmarker** model here as `face_landmarker.task`
(~3.6 MB). It is **not committed** (fetched on setup). `setup_linux.sh` downloads it for you, or grab
it manually:

```bash
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

The model is distributed by Google under the Apache-2.0 license.
