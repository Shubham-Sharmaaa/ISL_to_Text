// static/main.js
const videoElement = document.getElementById("video");
const canvasElement = document.getElementById("overlay");
const canvasCtx = canvasElement.getContext("2d");
const wordEl = document.getElementById("word");
const sentEl = document.getElementById("sentence");
const statusEl = document.getElementById("status");
const confRange = document.getElementById("conf");
const confVal = document.getElementById("conf-val");

let sentence = "";
let lastLabel = null;
let stableCount = 0;
const STABLE_N = 6; // hold frames to confirm
let CONF_THRESHOLD = parseFloat(confRange.value);

confRange.addEventListener("input", () => {
  CONF_THRESHOLD = parseFloat(confRange.value);
  confVal.textContent = CONF_THRESHOLD.toFixed(2);
});

document.getElementById("add-word").addEventListener("click", () => {
  sentence += wordEl.textContent;
  sentEl.textContent = sentence;
});
document.getElementById("clear-sentence").addEventListener("click", () => {
  sentence = "";
  sentEl.textContent = sentence;
});

// Setup MediaPipe Hands
const hands = new Hands({
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
  },
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6,
});

hands.onResults(onResults);

// Camera setup (MediaPipe camera utils)
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 640,
  height: 480,
});
camera.start();

// Resize canvas to match video
videoElement.addEventListener("loadeddata", () => {
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;
});

// Debounce network calls
let lastSent = 0;
const SEND_INTERVAL_MS = 80; // ~12.5 FPS

async function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const lm = results.multiHandLandmarks[0]; // 21 landmarks
    // draw landmarks
    drawConnectors(canvasCtx, lm, HAND_CONNECTIONS, {
      color: "#00FF00",
      lineWidth: 2,
    });
    drawLandmarks(canvasCtx, lm, { color: "#FF0000", lineWidth: 1 });

    // prepare landmark array as list of [x,y,z]
    const landmarks = lm.map((p) => [p.x, p.y, p.z]);

    const now = Date.now();
    if (now - lastSent > SEND_INTERVAL_MS) {
      lastSent = now;
      statusEl.textContent = "Sending landmarks...";
      try {
        const resp = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ landmarks: landmarks }),
        });
        const j = await resp.json();
        if (j.error) {
          statusEl.textContent = "Server error: " + j.error;
        } else {
          const label = j.label;
          const conf = parseFloat(j.confidence);
          statusEl.textContent = `Pred: ${label} (${conf.toFixed(2)})`;

          // debounce on client side: require STABLE_N in a row
          if (conf >= CONF_THRESHOLD) {
            if (label === lastLabel) {
              stableCount++;
            } else {
              stableCount = 1;
              lastLabel = label;
            }
            if (stableCount >= STABLE_N) {
              // apply action
              if (label === "SPACE") {
                sentence += " ";
              } else if (label === "DEL") {
                sentence = sentence.slice(0, -1);
              } else {
                // treat label as single char or word
                wordEl.textContent = label;
              }
              sentEl.textContent = sentence;
              stableCount = 0;
            }
          } else {
            stableCount = 0;
            lastLabel = null;
          }
        }
      } catch (err) {
        statusEl.textContent = "Network error";
      }
    }
  } else {
    // no hand
    statusEl.textContent = "No hand detected";
    // clear drawings (done by clearRect)
  }

  canvasCtx.restore();
}
