const BACKEND_URL = "/predict";

const videoEl = document.getElementById("video");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");

const currentLetterEl = document.getElementById("current-letter");
const sentenceEl = document.getElementById("sentence");
const statusEl = document.getElementById("status-text");

const addWordBtn = document.getElementById("add-word");
const clearSentenceBtn = document.getElementById("clear-sentence");

const confSlider = document.getElementById("conf-slider");
const confValueEl = document.getElementById("conf-value");

let sentence = "";
let lastLabel = null;
let stableCount = 0;
const STABLE_N = 7;

let CONF_THRESHOLD = parseFloat(confSlider.value);
confValueEl.textContent = CONF_THRESHOLD.toFixed(2);

// ------------ UI events ------------
confSlider.addEventListener("input", () => {
  CONF_THRESHOLD = parseFloat(confSlider.value);
  confValueEl.textContent = CONF_THRESHOLD.toFixed(2);
});

addWordBtn.addEventListener("click", () => {
  const letter = currentLetterEl.textContent;
  if (letter && letter !== "-") {
    sentence += letter;
    sentenceEl.textContent = sentence;
  }
});

clearSentenceBtn.addEventListener("click", () => {
  sentence = "";
  sentenceEl.textContent = sentence;
});

// ------------ Camera setup ------------
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    statusEl.textContent = "getUserMedia not supported in this browser.";
    throw new Error("getUserMedia not supported");
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    });
    videoEl.srcObject = stream;
    return new Promise((resolve) => {
      videoEl.onloadedmetadata = () => {
        videoEl.play();
        canvasEl.width = videoEl.videoWidth || 640;
        canvasEl.height = videoEl.videoHeight || 480;
        resolve();
      };
    });
  } catch (err) {
    console.error("Camera error:", err);
    statusEl.textContent = "Camera error: " + err.name;
    throw err;
  }
}

// ------------ MediaPipe Hands ------------
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6,
});

hands.onResults(onResults);

let lastSent = 0;
const SEND_INTERVAL_MS = 80;

async function frameLoop() {
  await hands.send({ image: videoEl });
  requestAnimationFrame(frameLoop);
}

// ------------ Main callback ------------
async function onResults(results) {
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  // Mirror display only (like cv2.flip)
  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-canvasEl.width, 0);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const lm = results.multiHandLandmarks[0];

    drawConnectors(ctx, lm, HAND_CONNECTIONS, {
      color: "#00FF00",
      lineWidth: 2,
    });
    drawLandmarks(ctx, lm, { color: "#FF0000", lineWidth: 1 });

    // Raw landmarks in original coordinate system
    const landmarks = lm.map((p) => [p.x, p.y, p.z]);

    const now = Date.now();
    if (now - lastSent > SEND_INTERVAL_MS) {
      lastSent = now;
      statusEl.textContent = "Sending landmarks...";

      try {
        const resp = await fetch(BACKEND_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ landmarks }),
        });

        if (!resp.ok) {
          statusEl.textContent = "HTTP error: " + resp.status;
          console.error("HTTP error:", resp.status);
          ctx.restore();
          return;
        }

        const data = await resp.json();
        console.log("Prediction from backend:", data);

        if (data.error) {
          statusEl.textContent = "Server error: " + data.error;
        } else {
          const label = data.label;
          const conf = parseFloat(data.confidence);

          statusEl.textContent = `Pred: ${label} (${conf.toFixed(2)})`;

          if (conf >= CONF_THRESHOLD) {
            if (label === lastLabel) {
              stableCount++;
            } else {
              stableCount = 1;
              lastLabel = label;
            }

            if (stableCount >= STABLE_N) {
              if (label === "SPACE") {
                sentence += " ";
              } else if (label === "DEL") {
                sentence = sentence.slice(0, -1);
              } else {
                currentLetterEl.textContent = label;
                sentence += label; // behave like infer_realtime
              }
              sentenceEl.textContent = sentence;
              stableCount = 0;
            }
          } else {
            stableCount = 0;
            lastLabel = null;
          }
        }
      } catch (err) {
        console.error("Network error:", err);
        statusEl.textContent = "Network error (see console)";
      }
    }
  } else {
    statusEl.textContent = "No hand detected";
    stableCount = 0;
    lastLabel = null;
  }

  ctx.restore();
}

// ------------ Bootstrap ------------
(async function start() {
  statusEl.textContent = "Requesting camera...";
  try {
    await setupCamera();
    statusEl.textContent = "Camera ready. Show a gesture.";
    requestAnimationFrame(frameLoop);
  } catch (e) {
    console.log("Error setting up camera:", e);
  }
})();
