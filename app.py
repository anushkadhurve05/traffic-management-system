from flask import Flask, Response, jsonify, render_template_string, request
from ultralytics import YOLO
import cv2
import threading
import time
import os
from werkzeug.utils import secure_filename


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# YOLO11 MODEL
# ============================================================

# Only YOLO11 is used in the final Phase 4 system.
model = YOLO("yolo11n.pt")


# ============================================================
# VIDEO SETTINGS
# ============================================================

UPLOAD_FOLDER = "traffic_uploads"

ALLOWED_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv"
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# VIDEO VARIABLES
# ============================================================

camera = None

video_path = None

source_changed = False

current_frame = None

video_name = "No video selected"


# ============================================================
# TRAFFIC VARIABLES
# ============================================================

vehicle_count = 0

traffic_level = "LOW"

signal_color = "GREEN"

signal_time = 30


# ============================================================
# THREAD LOCK
# ============================================================

lock = threading.Lock()


# ============================================================
# VEHICLE CLASSES
# ============================================================

# These are the vehicle classes available in YOLO11 COCO.

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
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# TRAFFIC LEVEL CALCULATION
# ============================================================

def calculate_traffic(count):

    if count <= 4:
        return "LOW"

    elif count <= 9:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# ADAPTIVE SIGNAL CALCULATION
# ============================================================

def calculate_signal(traffic):

    # The system dynamically gives more green time
    # when more vehicles are detected.

    if traffic == "LOW":

        return "GREEN", 30

    elif traffic == "MEDIUM":

        return "GREEN", 45

    else:

        return "GREEN", 60


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

    local_path = None

    while True:

        # ----------------------------------------------------
        # CHECK FOR NEW VIDEO
        # ----------------------------------------------------

        with lock:

            requested_path = video_path

            changed = source_changed

            if changed:

                source_changed = False


        # ----------------------------------------------------
        # OPEN / REOPEN VIDEO
        # ----------------------------------------------------

        if requested_path != local_path or changed:

            if camera is not None:

                camera.release()

                camera = None


            local_path = requested_path


            if local_path is not None:

                camera = cv2.VideoCapture(local_path)


                if not camera.isOpened():

                    print(
                        "ERROR: Could not open video:",
                        local_path
                    )

                    camera = None

                else:

                    print(
                        "AI processing video:",
                        local_path
                    )


        # ----------------------------------------------------
        # NO VIDEO
        # ----------------------------------------------------

        if camera is None:

            time.sleep(0.1)

            continue


        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        success, frame = camera.read()


        # ----------------------------------------------------
        # VIDEO FINISHED
        # ----------------------------------------------------

        if not success:

            # Restart video automatically.

            camera.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            time.sleep(0.05)

            continue


        try:

            # =================================================
            # YOLO11 DETECTION
            # =================================================

            results = model(
                frame,
                conf=0.35,
                verbose=False
            )


            # =================================================
            # VEHICLE COUNT
            # =================================================

            count = 0


            if results[0].boxes is not None:

                for box in results[0].boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = model.names[
                        class_id
                    ]


                    if class_name in vehicle_classes:

                        count += 1


            # =================================================
            # CREATE ANNOTATED FRAME
            # =================================================

            annotated_frame = results[0].plot()


            # =================================================
            # CALCULATE TRAFFIC LEVEL
            # =================================================

            level = calculate_traffic(
                count
            )


            # =================================================
            # CALCULATE ADAPTIVE SIGNAL
            # =================================================

            color, duration = calculate_signal(
                level
            )


            # =================================================
            # UPDATE GLOBAL DATA
            # =================================================

            with lock:

                vehicle_count = count

                traffic_level = level

                signal_color = color

                signal_time = duration


            # =================================================
            # DISPLAY AI INFORMATION
            # =================================================

            cv2.putText(
                annotated_frame,
                f"Vehicles Detected: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )


            cv2.putText(
                annotated_frame,
                f"Traffic Level: {level}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )


            cv2.putText(
                annotated_frame,
                f"Adaptive Green Time: {duration} sec",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


            cv2.putText(
                annotated_frame,
                "AI Traffic Management Active",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            # =================================================
            # SIGNAL INDICATOR
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
                2
            )


            # =================================================
            # ENCODE FRAME AS JPEG
            # =================================================

            ret, buffer = cv2.imencode(
                ".jpg",
                annotated_frame
            )


            if ret:

                with lock:

                    current_frame = (
                        buffer.tobytes()
                    )


        except Exception as e:

            print(
                "AI Processing Error:",
                e
            )


        # Small delay to control CPU usage.

        time.sleep(0.03)


# ============================================================
# VIDEO STREAM GENERATOR
# ============================================================

def generate_frames():

    global current_frame

    while True:

        with lock:

            frame = current_frame


        if frame is not None:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )


        time.sleep(0.03)


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


        /* =================================================
           HEADER
        ================================================= */

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


        /* =================================================
           MAIN CONTAINER
        ================================================= */

        .container {

            width:
                92%;

            max-width:
                1200px;

            margin:
                25px auto;
        }


        /* =================================================
           GRID
        ================================================= */

        .grid {

            display:
                grid;

            grid-template-columns:
                1fr 1fr;

            gap:
                20px;
        }


        /* =================================================
           CARD
        ================================================= */

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


        /* =================================================
           UPLOAD
        ================================================= */

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


        /* =================================================
           STATUS TEXT
        ================================================= */

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


        /* =================================================
           AI VIDEO
        ================================================= */

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


        /* =================================================
           STATUS GRID
        ================================================= */

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


        /* =================================================
           TRAFFIC BADGES
        ================================================= */

        .traffic-badge {

            display:
                inline-block;

            margin-top:
                8px;

            padding:
                6px 12px;

            border-radius:
                20px;

            font-size:
                13px;

            font-weight:
                bold;
        }


        /* =================================================
           SIGNAL
        ================================================= */

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

            background:
                green;
        }


        .yellow {

            background:
                yellow;
        }


        .red {

            background:
                red;
        }


        /* =================================================
           RESPONSIVE
        ================================================= */

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


<!-- =====================================================
     HEADER
===================================================== -->

<div class="header">

    <h1>
        Smart Traffic Management System
    </h1>

    <p>
        AI-Based Adaptive Traffic Signal Management
    </p>

</div>


<div class="container">


    <!-- =================================================
         TOP GRID
    ================================================= -->

    <div class="grid">


        <!-- =============================================
             VIDEO UPLOAD CARD
        ============================================== -->

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
                Upload a traffic video to start
                AI analysis.
            </div>

        </div>


        <!-- =============================================
             AI VIDEO CARD
        ============================================== -->

        <div class="card">

            <h2>
                AI Traffic Detection
            </h2>


            <img
                class="camera"
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


    <!-- =================================================
         AI STATUS
    ================================================= -->

    <div
        class="card"
        style="margin-top:20px;"
    >

        <h2>
            AI Traffic Status
        </h2>


        <div class="status">


            <!-- VEHICLES -->

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


            <!-- TRAFFIC -->

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


            <!-- SIGNAL -->

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


            <!-- TIME -->

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


    <!-- =================================================
         HOW SYSTEM WORKS
    ================================================= -->

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
            4. The system counts cars, buses, trucks
            and motorcycles.
        </p>


        <p>
            5. The vehicle count determines the
            traffic level.
        </p>


        <p>
            6. The system automatically calculates
            adaptive green time.
        </p>


        <p>
            7. The result is displayed on the website.
        </p>

    </div>


</div>


<!-- =====================================================
     JAVASCRIPT
===================================================== -->

<script>


// ========================================================
// UPLOAD VIDEO
// ========================================================

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


    .then(response =>
        response.json()
    )


    .then(data => {

        if (data.success) {

            status.innerText =
                "Video uploaded successfully.";

            videoName.innerText =
                "Processing: " +
                data.filename;

            decision.innerText =
                "YOLO11 is analyzing the traffic video.";

        }

        else {

            status.innerText =
                data.message;

        }

    })


    .catch(error => {

        console.error(error);

        status.innerText =
            "Upload failed. Check the Flask terminal.";

    });

}


// ========================================================
// UPDATE DASHBOARD
// ========================================================

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

        signal.classList.add(
            "green"
        );

    }

    else if (
        data.signal_color === "YELLOW"
    ) {

        signal.classList.add(
            "yellow"
        );

    }

    else {

        signal.classList.add(
            "red"
        );

    }


    // Update decision message.

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


    document.getElementById(
        "decision"
    ).innerText =
        message;

}


// ========================================================
// CLEAR SYSTEM
// ========================================================

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

    })


    .catch(error => {

        console.error(error);

    });

}


// ========================================================
// LIVE STATUS UPDATE
// ========================================================

function updateLiveData() {

    fetch(
        "/status"
    )


    .then(response =>
        response.json()
    )


    .then(data => {

        updateDashboard(data);


        if (
            data.video_name &&
            data.video_name !== "No video selected"
        ) {

            document.getElementById(
                "videoName"
            ).innerText =
                "Processing: " +
                data.video_name;

        }

    })


    .catch(error => {

        console.error(
            "Status update error:",
            error
        );

    });

}


// Update dashboard every second.

setInterval(
    updateLiveData,
    1000
);


</script>


</body>

</html>

"""


# ============================================================
# HOME PAGE
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

    return Response(
        generate_frames(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# VIDEO UPLOAD API
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


    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CHECK FORMAT
    # --------------------------------------------------------

    if not allowed_file(
        file.filename
    ):

        return jsonify({

            "success": False,

            "message":
                "Unsupported video format. "
                "Use MP4, AVI, MOV, or MKV."

        })


    # --------------------------------------------------------
    # CREATE SAFE FILE NAME
    # --------------------------------------------------------

    filename =secure_filename(
            file.filename
        )


    timestamp = str(
            int(
                 time.time()
            )
        )


    filename =timestamp + "_" + filename


    path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )


    # --------------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------------

    file.save(path)


    # --------------------------------------------------------
    # UPDATE SYSTEM
    # --------------------------------------------------------

    with lock:

        video_path = path

        video_name = filename

        source_changed = True

        vehicle_count = 0

        traffic_level = "LOW"

        signal_color = "GREEN"

        signal_time = 30

        current_frame = None


    print(
        "New traffic video uploaded:",
        path
    )


    return jsonify({

        "success": True,

        "filename": filename

    })


# ============================================================
# CLEAR SYSTEM API
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


    with lock:

        # Release current video.

        if camera is not None:

            camera.release()

            camera = None


        video_path = None

        source_changed = True

        vehicle_count = 0

        traffic_level = "LOW"

        signal_color = "GREEN"

        signal_time = 30

        current_frame = None

        video_name = "No video selected"


    return jsonify({

        "vehicle_count": 0,

        "traffic_level": "LOW",

        "signal_color": "GREEN",

        "signal_time": 30,

        "video_name":
            "No video selected"

    })


# ============================================================
# STATUS API
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
                video_name

        })


# ============================================================
# START AI THREAD
# ============================================================

ai_thread = threading.Thread(
    target=camera_loop,
    daemon=True
)


ai_thread.start()


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print(
        "======================================"
    )
    print(
        " SMART TRAFFIC MANAGEMENT SYSTEM"
    )
    print(
        "======================================"
    )
    print("")

    print(
        "AI Model: YOLO11"
    )

    print(
        "AI Function: Vehicle Detection"
    )

    print(
        "Adaptive Traffic Signal: ENABLED"
    )

    print("")

    print(
        "Open this in your browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("")

    print(
        "Upload a traffic video from the webpage."
    )

    print(
        "YOLO11 will analyze the uploaded video."
    )

    print("")


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        threaded=True

    )