# Science

This application leverages an NVIDIA XAVIER (or comparable edge hardware) to perform real-time object detection and zooming using a PTZ (Pan‐Tilt‐Zoom) camera. By automatically detecting and focusing on objects such as deer, other wildlife, or notable scene elements, the system provides key insights for ecological monitoring, wildlife management, and situational awareness. Its scientific relevance lies in continuous, unobtrusive observation of dynamic environments—helping researchers and practitioners gather high-quality images for biodiversity studies, behavior research, and timely interventions in remote or hard-to-reach locations.

# AI@Edge

The application deploys the **Florence v2** base model on the edge device. Florence v2 is a powerful vision-language model capable of identifying a variety of objects and scenes. The workflow is:

1. The camera rotates (pan/tilt) and zooms in pre-determined or incremental steps to scan the environment.  
2. Live frames are captured and processed by the Florence v2 model.  
3. If an “interesting” object or scene is detected, the system automatically adjusts the PTZ camera to center and maximize the object in the frame.  
4. A picture is taken and sent to the Sage cloud infrastructure for further processing, archiving, or real-time alerts.

By pushing this AI capability to the edge, the system operates continuously with minimal latency and reduced bandwidth usage—uploading only relevant snapshots rather than a constant video feed.

# Arguments
The application supports the following command-line arguments:

- **`--iterations` / `-it`**  
 Number of PTZ camera rounds to run (Default: 5)

- **`--objects` / `-obj`**  
 Objects to detect (comma-separated). Use "*" to detect all objects. (Default: "person")

- **`--username` / `-un`**  
 Username for the PTZ camera

- **`--password` / `-pw`**  
 Password for the PTZ camera

- **`--cameraip` / `-ip`**  
 IP address of the PTZ camera

- **`--panstep` / `-ps`**  
 Pan step in degrees (Default: 15)

- **`--tilt` / `-tv`**  
 Tilt value in degrees (Default: 0)

- **`--zoom` / `-zm`**  
 Zoom value (Default: 1)

- **`--model` / `-m`**  
 Detection model to use: "yolo11n" or "Florence-base" (Default: "yolo11n")

- **`--iterdelay` / `-id`**  
 Delay between iterations in seconds (Default: 60.0)

- **`--confidence` / `-conf`**  
 Minimum confidence threshold for detections (0-1) (Default: 0.1)

Example usage:
```bash
python main.py -it 10 -obj "person,car" -un admin -pw secret -ip xxx.xxx.x.xx -m yolo11n -conf 0.1

# Ontology

The interesting images collected by the system are tagged with metadata for easy retrieval and analysis.
