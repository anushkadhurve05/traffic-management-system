from flask import Flask, Response, jsonify, render_template_string, request
from ultralytics import YOLO
import cv2
import threading
import time
import os
import numpy as np
from werkzeug.utils import secure_filename


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# VIDEO SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "traffic_uploads"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo11n.pt"
)

ALLOWED_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv"
}

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# YOLO11 MODEL
# ============================================================

print("Loading YOLO11 model...")

model = YOLO(MODEL_PATH)

print("YOLO11 model loaded successfully.")


# ============================================================
# VIDEO VARIABLES
# ============================================================

camera = None

video_path = None

source_changed = False

current_frame = None

video_name = "No video selected"

video_status = "Waiting for video upload"


# ============================================================
# TRAFFIC VARIABLES
# ============================================================

vehicle_count = 0

traffic_level = "LOW"

signal_color = "GREEN"

signal_time = 30


# ============================================================
# THREAD CONTROL
# ============================================================

lock = threading.Lock()

stop_processing = False


# ============================================================
# VEHICLE CLASSES
# ============================================================

vehicle_classes = [
    "car",
    "bus",
    "truck",
    "motorcycle"
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower() in ALLOWED_EXTENSIONS
    )


# ============================================================
# TRAFFIC LEVEL
# ============================================================

def calculate_traffic(count):

    if count <= 4:
        return "LOW"

    elif count <= 9:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# ADAPTIVE SIGNAL
# ============================================================

def calculate_signal(traffic):

    if traffic == "LOW":

        return "GREEN", 30

    elif traffic == "MEDIUM":

        return "GREEN", 45

    else:

        return "GREEN", 60


# ============================================================
# WAITING FRAME
# ============================================================

def create_waiting_frame(message):

    frame = np.zeros(
        (480, 854, 3),
        dtype=np.uint8
    )

    cv2.putText(
        frame,
        "SMART TRAFFIC MANAGEMENT SYSTEM",
        (80, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        message,
        (130, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "YOLO11 AI Traffic Detection",
        (210, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA
    )

    success, buffer = cv2.imencode(
        ".jpg",
        frame
    )

    if success:

        return buffer.tobytes()

    return None


# ============================================================
# AI VIDEO PROCESSING
# ============================================================

def camera_loop():

    global camera
    global video_path
    global source_changed
    global vehicle_count
    global traffic_level
    global signal_color
    global signal_time
    global current_frame
    global video_status

    local_path = None

    print("AI processing thread started.")

    while True:

        try:

            # =================================================
            # CHECK VIDEO STATE
            # =================================================

            with lock:

                requested_path = video_path

                changed = source_changed

                if changed:
                    source_changed = False


            # =================================================
            # NEW VIDEO
            # =================================================

            if requested_path != local_path or changed:

                # Release old capture safely.

                if camera is not None:

                    try:
                        camera.release()
                    except Exception:
                        pass

                    camera = None


                local_path = requested_path


                if local_path is None:

                    with lock:

                        video_status = (
                            "Waiting for video upload."
                        )

                    time.sleep(0.1)

                    continue


                local_path = os.path.abspath(
                    local_path
                )


                # =================================================
                # CHECK FILE
                # =================================================

                if not os.path.exists(local_path):

                    print(
                        "ERROR: Video file does not exist:",
                        local_path
                    )

                    with lock:

                        video_status = (
                            "ERROR: Uploaded video file not found."
                        )

                    local_path = None

                    time.sleep(0.5)

                    continue


                file_size = os.path.getsize(
                    local_path
                )

                print(
                    "======================================"
                )

                print(
                    "Opening uploaded video:"
                )

                print(
                    local_path
                )

                print(
                    "Video size:",
                    file_size,
                    "bytes"
                )


                if file_size <= 0:

                    print(
                        "ERROR: Video file is empty."
                    )

                    with lock:

                        video_status = (
                            "ERROR: Uploaded video is empty."
                        )

                    local_path = None

                    time.sleep(0.5)

                    continue


                # =================================================
                # OPEN VIDEO
                # =================================================

                new_camera = cv2.VideoCapture(
                    local_path
                )


                if not new_camera.isOpened():

                    print(
                        "ERROR: OpenCV could not open video."
                    )

                    with lock:

                        video_status = (
                            "ERROR: OpenCV could not decode this video."
                        )

                    new_camera.release()

                    camera = None

                    local_path = None

                    time.sleep(0.5)

                    continue


                width = int(
                    new_camera.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                height = int(
                    new_camera.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )

                fps = new_camera.get(
                    cv2.CAP_PROP_FPS
                )


                print(
                    "Video opened successfully."
                )

                print(
                    "Resolution:",
                    width,
                    "x",
                    height
                )

                print(
                    "FPS:",
                    fps
                )


                camera = new_camera


                with lock:

                    video_status = (
                        "Video opened. "
                        "YOLO11 AI processing started."
                    )


            # =================================================
            # NO CAMERA
            # =================================================

            if camera is None:

                time.sleep(0.1)

                continue


            # =================================================
            # READ FRAME
            # =================================================

            success, frame = camera.read()


            # =================================================
            # END OF VIDEO
            # =================================================

            if not success:

                print(
                    "Video reached end. Restarting..."
                )


                camera.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )


                with lock:

                    video_status = (
                        "Video reached the end. "
                        "Restarting video..."
                    )


                time.sleep(0.1)

                continue


            # =================================================
            # AI PROCESSING
            # =================================================

            with lock:

                video_status = (
                    "YOLO11 is processing video frames."
                )


            results = model(
                frame,
                conf=0.35,
                verbose=False
            )


            # =================================================
            # VEHICLE COUNT
            # =================================================

            count = 0


            if (
                results
                and
                results[0].boxes is not None
            ):

                for box in results[0].boxes:

                    try:

                        class_id = int(
                            box.cls[0]
                        )

                        class_name = model.names[
                            class_id
                        ]

                        if class_name in vehicle_classes:

                            count += 1

                    except Exception as e:

                        print(
                            "Detection parsing error:",
                            repr(e)
                        )


            # =================================================
            # ANNOTATED FRAME
            # =================================================

            annotated_frame = results[0].plot()


            # =================================================
            # TRAFFIC
            # =================================================

            level = calculate_traffic(
                count
            )


            color, duration = calculate_signal(
                level
            )


            # =================================================
            # UPDATE GLOBAL STATUS
            # =================================================

            with lock:

                vehicle_count = count

                traffic_level = level

                signal_color = color

                signal_time = duration


            # =================================================
            # INFORMATION ON VIDEO
            # =================================================

            cv2.putText(
                annotated_frame,
                f"Vehicles Detected: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                annotated_frame,
                f"Traffic Level: {level}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                annotated_frame,
                f"Adaptive Green Time: {duration} sec",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            cv2.putText(
                annotated_frame,
                "AI Traffic Management Active",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            # =================================================
            # SIGNAL COLOR
            # =================================================

            if color == "GREEN":

                signal_bgr = (0, 200, 0)

            elif color == "YELLOW":

                signal_bgr = (0, 255, 255)

            else:

                signal_bgr = (0, 0, 255)


            cv2.circle(
                annotated_frame,
                (50, 220),
                20,
                signal_bgr,
                -1
            )


            cv2.putText(
                annotated_frame,
                f"Signal: {color}",
                (85, 228),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            # =================================================
            # JPEG ENCODING
            # =================================================

            success, buffer = cv2.imencode(
                ".jpg",
                annotated_frame,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    80
                ]
            )


            if not success:

                print(
                    "ERROR: Could not encode frame."
                )

                continue


            frame_bytes = buffer.tobytes()


            # =================================================
            # STORE FRAME
            # =================================================

            with lock:

                current_frame = frame_bytes


            # =================================================
            # SMALL CPU DELAY
            # =================================================

            time.sleep(0.01)


        except Exception as e:

            print(
                "======================================"
            )

            print(
                "AI PROCESSING ERROR:"
            )

            print(
                repr(e)
            )

            print(
                "======================================"
            )


            with lock:

                video_status = (
                    "AI processing error: "
                    + str(e)[:180]
                )


            time.sleep(0.5)


# ============================================================
# VIDEO STREAM GENERATOR
# ============================================================

def generate_frames():

    waiting_frame = create_waiting_frame(
        "Waiting for video upload..."
    )

    while True:

        with lock:

            frame = current_frame

            status = video_status


        # =====================================================
        # USE ACTUAL AI FRAME
        # =====================================================

        if frame is not None:

            output_frame = frame


        # =====================================================
        # OTHERWISE SHOW STATUS FRAME
        # =====================================================

        else:

            output_frame = create_waiting_frame(
                status
            )


        if output_frame is not None:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(
                    len(output_frame)
                ).encode()
                + b"\r\n"
                b"Cache-Control: no-cache\r\n"
                b"\r\n"
                + output_frame
                + b"\r\n"
            )


        time.sleep(0.05)


# ============================================================
# WEBSITE
# ============================================================

HTML_PAGE = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
Smart Traffic Management System
</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        #f2f5f8;

    color:
        #1f2937;
}

.header {

    background:
        #1f2937;

    color:
        white;

    padding:
        25px;

    text-align:
        center;
}

.header h1 {

    margin:
        0;

    font-size:
        30px;
}

.header p {

    margin:
        8px 0 0;

    font-size:
        16px;
}

.container {

    width:
        92%;

    max-width:
        1200px;

    margin:
        25px auto;
}

.grid {

    display:
        grid;

    grid-template-columns:
        1fr 1fr;

    gap:
        20px;
}

.card {

    background:
        white;

    padding:
        20px;

    border-radius:
        12px;

    box-shadow:
        0 3px 10px
        rgba(0, 0, 0, 0.10);
}

.card h2 {

    margin-top:
        0;

    margin-bottom:
        18px;
}

label {

    display:
        block;

    margin-top:
        15px;

    font-weight:
        bold;
}

input[type="file"] {

    width:
        100%;

    padding:
        10px;

    margin-top:
        8px;

    border:
        1px solid #ccc;

    border-radius:
        6px;
}

button {

    padding:
        12px 20px;

    border:
        none;

    border-radius:
        6px;

    cursor:
        pointer;

    font-weight:
        bold;

    margin-top:
        15px;
}

.upload {

    width:
        100%;

    background:
        #16a34a;

    color:
        white;
}

.clear {

    background:
        #e5e7eb;

    margin-left:
        8px;
}

.upload-status {

    margin-top:
        10px;

    text-align:
        center;

    font-weight:
        bold;
}

.decision {

    margin-top:
        20px;

    padding:
        15px;

    border-radius:
        8px;

    background:
        #eef2ff;

    text-align:
        center;

    font-weight:
        bold;

    line-height:
        1.5;
}

.camera {

    width:
        100%;

    min-height:
        300px;

    object-fit:
        contain;

    background:
        black;

    border-radius:
        10px;
}

.video-name {

    margin-top:
        10px;

    font-size:
        14px;

    color:
        #374151;

    word-break:
        break-word;
}

.status {

    display:
        grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap:
        15px;
}

.status-box {

    padding:
        20px;

    border-radius:
        10px;

    text-align:
        center;

    background:
        #f3f4f6;
}

.status-box h3 {

    margin:
        0 0 10px;

    font-size:
        16px;
}

.value {

    font-size:
        28px;

    font-weight:
        bold;
}

.signal-wrapper {

    display:
        flex;

    justify-content:
        center;

    align-items:
        center;

    min-height:
        90px;
}

.signal {

    width:
        65px;

    height:
        65px;

    border-radius:
        50%;

    border:
        5px solid #333;
}

.green {
    background: green;
}

.yellow {
    background: yellow;
}

.red {
    background: red;
}

@media(max-width: 900px) {

    .grid {

        grid-template-columns:
            1fr;
    }

    .status {

        grid-template-columns:
            repeat(2, 1fr);
    }
}

@media(max-width: 500px) {

    .status {

        grid-template-columns:
            1fr;
    }

    .header h1 {

        font-size:
            24px;
    }
}

</style>

</head>

<body>

<div class="header">

<h1>
Smart Traffic Management System
</h1>

<p>
AI-Based Adaptive Traffic Signal Management
</p>

</div>


<div class="container">

<div class="grid">


<div class="card">

<h2>
Traffic Video Input
</h2>

<label>
Select Traffic Video
</label>

<input
    type="file"
    id="videoFile"
    accept=".mp4,.avi,.mov,.mkv"
>

<button
    class="upload"
    onclick="uploadVideo()"
>
Upload & Analyze Traffic
</button>

<button
    class="clear"
    onclick="clearSystem()"
>
Clear
</button>

<div
    class="upload-status"
    id="uploadStatus"
>
No video selected
</div>

<div
    class="decision"
    id="decision"
>
Upload a traffic video to start AI analysis.
</div>

</div>


<div class="card">

<h2>
AI Traffic Detection
</h2>

<img
    class="camera"
    id="trafficVideo"
    src="/video_feed"
    alt="AI processed traffic video"
>

<div
    class="video-name"
    id="videoName"
>
AI processed video will appear here.
</div>

</div>

</div>


<div
    class="card"
    style="margin-top:20px;"
>

<h2>
AI Traffic Status
</h2>

<div class="status">

<div class="status-box">

<h3>
Vehicles Detected
</h3>

<div
    class="value"
    id="vehicles"
>
0
</div>

</div>


<div class="status-box">

<h3>
Traffic Level
</h3>

<div
    class="value"
    id="traffic"
>
LOW
</div>

</div>


<div class="status-box">

<h3>
Signal
</h3>

<div class="signal-wrapper">

<div
    id="signal"
    class="signal green"
>
</div>

</div>

<div
    id="signalText"
    style="font-weight:bold;"
>
GREEN
</div>

</div>


<div class="status-box">

<h3>
Adaptive Green Time
</h3>

<div
    class="value"
    id="time"
>
30 s
</div>

</div>

</div>

</div>


<div
    class="card"
    style="margin-top:20px;"
>

<h2>
How the AI System Works
</h2>

<p>
1. The user uploads a traffic video.
</p>

<p>
2. OpenCV reads the video frame by frame.
</p>

<p>
3. YOLO11 detects vehicles in each frame.
</p>

<p>
4. The system counts cars, buses, trucks and motorcycles.
</p>

<p>
5. The vehicle count determines the traffic level.
</p>

<p>
6. The system automatically calculates adaptive green time.
</p>

<p>
7. The result is displayed on the website.
</p>

</div>

</div>


<script>

function uploadVideo() {

    const fileInput =
        document.getElementById("videoFile");

    const status =
        document.getElementById("uploadStatus");

    const decision =
        document.getElementById("decision");

    const videoName =
        document.getElementById("videoName");

    if (fileInput.files.length === 0) {

        status.innerText =
            "Please select a traffic video first.";

        return;
    }

    const file =
        fileInput.files[0];

    const formData =
        new FormData();

    formData.append(
        "video",
        file
    );

    status.innerText =
        "Uploading video...";

    decision.innerText =
        "Please wait. AI analysis is starting...";

    fetch(
        "/upload_video",
        {
            method: "POST",
            body: formData
        }
    )

    .then(response => response.json())

    .then(data => {

        if (data.success) {

            status.innerText =
                "Video uploaded successfully.";

            videoName.innerText =
                "Processing: " +
                data.filename;

            decision.innerText =
                "YOLO11 is analyzing the traffic video.";

            // Force the browser to reconnect to the
            // MJPEG stream after a new upload.

            const video =
                document.getElementById(
                    "trafficVideo"
                );

            video.src =
                "/video_feed?t=" +
                Date.now();

        }

        else {

            status.innerText =
                data.message;

        }

    })

    .catch(error => {

        console.error(error);

        status.innerText =
            "Upload failed. Check Render logs.";

    });

}


function updateDashboard(data) {

    document.getElementById(
        "vehicles"
    ).innerText =
        data.vehicle_count;


    document.getElementById(
        "traffic"
    ).innerText =
        data.traffic_level;


    document.getElementById(
        "time"
    ).innerText =
        data.signal_time + " s";


    document.getElementById(
        "signalText"
    ).innerText =
        data.signal_color;


    const signal =
        document.getElementById(
            "signal"
        );


    signal.className =
        "signal";


    if (
        data.signal_color === "GREEN"
    ) {

        signal.classList.add("green");

    }

    else if (
        data.signal_color === "YELLOW"
    ) {

        signal.classList.add("yellow");

    }

    else {

        signal.classList.add("red");

    }


    let message =
        "AI detected " +
        data.vehicle_count +
        " vehicles. ";


    message +=
        "Traffic level: " +
        data.traffic_level +
        ". ";


    message +=
        "Adaptive green time: " +
        data.signal_time +
        " seconds.";


    if (
        data.video_status
    ) {

        message +=
            " " +
            data.video_status;
    }


    document.getElementById(
        "decision"
    ).innerText =
        message;

}


function clearSystem() {

    const fileInput =
        document.getElementById(
            "videoFile"
        );

    fileInput.value = "";


    document.getElementById(
        "uploadStatus"
    ).innerText =
        "System ready for a new video.";


    document.getElementById(
        "videoName"
    ).innerText =
        "AI processed video will appear here.";


    document.getElementById(
        "decision"
    ).innerText =
        "Upload a traffic video to start AI analysis.";


    fetch(
        "/clear",
        {
            method: "POST"
        }
    )

    .then(response =>
        response.json()
    )

    .then(data => {

        updateDashboard(data);

        const video =
            document.getElementById(
                "trafficVideo"
            );

        video.src =
            "/video_feed?t=" +
            Date.now();

    })

    .catch(error => {

        console.error(error);

    });

}


function updateLiveData() {

    fetch(
        "/status?t=" +
        Date.now()
    )

    .then(response =>
        response.json()
    )

    .then(data => {

        updateDashboard(data);


        if (
            data.video_name &&
            data.video_name !==
            "No video selected"
        ) {

            document.getElementById(
                "videoName"
            ).innerText =
                "Processing: " +
                data.video_name;

        }


        if (
            data.video_status
        ) {

            document.getElementById(
                "uploadStatus"
            ).innerText =
                data.video_status;

        }

    })

    .catch(error => {

        console.error(
            "Status update error:",
            error
        );

    });

}


setInterval(
    updateLiveData,
    1000
);

</script>

</body>

</html>

"""


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template_string(
        HTML_PAGE
    )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    response = Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

    response.headers[
        "Cache-Control"
    ] = "no-cache, no-store, must-revalidate, max-age=0"

    response.headers[
        "Pragma"
    ] = "no-cache"

    response.headers[
        "Expires"
    ] = "0"

    response.headers[
        "X-Accel-Buffering"
    ] = "no"

    response.headers[
        "Connection"
    ] = "keep-alive"

    return response


# ============================================================
# UPLOAD VIDEO
# ============================================================

@app.route(
    "/upload_video",
    methods=["POST"]
)
def upload_video():

    global video_path
    global source_changed
    global video_name
    global vehicle_count
    global traffic_level
    global signal_color
    global signal_time
    global current_frame
    global video_status


    if "video" not in request.files:

        return jsonify({

            "success": False,

            "message":
                "No video file was selected."

        })


    file = request.files["video"]


    if file.filename == "":

        return jsonify({

            "success": False,

            "message":
                "No video file was selected."

        })


    if not allowed_file(
        file.filename
    ):

        return jsonify({

            "success": False,

            "message":
                "Unsupported video format. "
                "Use MP4, AVI, MOV, or MKV."

        })


    # ========================================================
    # SAFE FILE NAME
    # ========================================================

    filename = secure_filename(
        file.filename
    )


    timestamp = str(
        int(
            time.time()
        )
    )


    filename = (
        timestamp
        + "_"
        + filename
    )


    path = os.path.abspath(
        os.path.join(
            UPLOAD_FOLDER,
            filename
        )
    )


    # ========================================================
    # SAVE
    # ========================================================

    try:

        file.save(path)

    except Exception as e:

        print(
            "UPLOAD SAVE ERROR:",
            repr(e)
        )

        return jsonify({

            "success": False,

            "message":
                "Could not save uploaded video."

        })


    # ========================================================
    # VERIFY FILE
    # ========================================================

    if not os.path.exists(path):

        return jsonify({

            "success": False,

            "message":
                "Video upload failed."

        })


    file_size = os.path.getsize(
        path
    )


    print(
        "New traffic video uploaded:",
        path
    )

    print(
        "Uploaded file size:",
        file_size,
        "bytes"
    )


    if file_size == 0:

        return jsonify({

            "success": False,

            "message":
                "Uploaded video is empty."

        })


    # ========================================================
    # UPDATE STATE
    # ========================================================

    with lock:

        video_path = path

        video_name = filename

        source_changed = True

        vehicle_count = 0

        traffic_level = "LOW"

        signal_color = "GREEN"

        signal_time = 30

        current_frame = None

        video_status = (
            "Video uploaded. "
            "Starting YOLO11 AI processor..."
        )


    return jsonify({

        "success": True,

        "filename": filename

    })


# ============================================================
# CLEAR SYSTEM
# ============================================================

@app.route(
    "/clear",
    methods=["POST"]
)
def clear_system():

    global camera
    global video_path
    global source_changed
    global vehicle_count
    global traffic_level
    global signal_color
    global signal_time
    global current_frame
    global video_name
    global video_status


    with lock:

        if camera is not None:

            try:

                camera.release()

            except Exception:

                pass

            camera = None


        video_path = None

        source_changed = True

        vehicle_count = 0

        traffic_level = "LOW"

        signal_color = "GREEN"

        signal_time = 30

        current_frame = None

        video_name = (
            "No video selected"
        )

        video_status = (
            "System cleared. "
            "Waiting for video upload."
        )


    return jsonify({

        "vehicle_count": 0,

        "traffic_level": "LOW",

        "signal_color": "GREEN",

        "signal_time": 30,

        "video_name":
            "No video selected",

        "video_status":
            "System cleared. "
            "Waiting for video upload."

    })


# ============================================================
# STATUS
# ============================================================

@app.route("/status")
def status():

    with lock:

        return jsonify({

            "vehicle_count":
                vehicle_count,

            "traffic_level":
                traffic_level,

            "signal_color":
                signal_color,

            "signal_time":
                signal_time,

            "video_name":
                video_name,

            "video_status":
                video_status

        })


# ============================================================
# START AI THREAD
# ============================================================

ai_thread = threading.Thread(
    target=camera_loop,
    daemon=True,
    name="YOLO11-Traffic-Processor"
)

ai_thread.start()


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "======================================"
    )

    print(
        " SMART TRAFFIC MANAGEMENT SYSTEM"
    )

    print(
        "======================================"
    )

    print(
        "AI Model: YOLO11"
    )

    print(
        "AI Function: Vehicle Detection"
    )

    print(
        "Adaptive Traffic Signal: ENABLED"
    )

    print(
        "AI processing thread: RUNNING"
    )

    print(
        "======================================"
    )


    app.run(

        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),

        debug=False,

        threaded=True

    )
