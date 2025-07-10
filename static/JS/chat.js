const language = document.getElementById("session-lang").value;
const exitText = document.getElementById("exit-text");
const headerTitle = document.getElementById("header-title");
const toggleBtn = document.getElementById("toggle-camera-btn");
const nextBtn = document.getElementById("next-btn");
const startBtn = document.getElementById("start-session-btn");

let isRecording = false;
let hasStartedOnce = false;
let mediaRecorder;
let recordedChunks = [];

let audioStream;
let audioRecorder;
let audioChunks = [];

const videoEl = document.getElementById("user-video");

const labels = {
  loading: { ar: "جارٍ التحميل...", en: "Loading...", fr: "Chargement...", de: "Lade..." },
  next: { ar: "التالي ⬅", en: "⬅ Next", fr: "⬅ Suivant", de: "⬅ Weiter" },
  start: { ar: "ابدأ الجلسة", en: "Start Session", fr: "Démarrer la session", de: "Sitzung starten" },
  pause: { ar: "إيقاف مؤقت", en: "Pause", fr: "Pause", de: "Pause" },
  resume: { ar: "استئناف", en: "Resume", fr: "Reprendre", de: "Fortsetzen" }
};

function getLocalizedText(key) {
  return labels[key][language] || key;
}

document.getElementById("loading-text").textContent = getLocalizedText("loading");
nextBtn.textContent = getLocalizedText("next");
nextBtn.disabled = true;
startBtn.textContent = getLocalizedText("start");

if (language === "ar") {
  exitText.textContent = "خروج";
  headerTitle.textContent = "جلسة دعم مع يوكا";
  toggleBtn.textContent = "إخفاء / عرض الكاميرا";
} else if (language === "fr") {
  exitText.textContent = "Sortie";
  headerTitle.textContent = "Session de soutien avec Yoka";
  toggleBtn.textContent = "Afficher / Masquer la caméra";
} else if (language === "de") {
  exitText.textContent = "Ausgang";
  headerTitle.textContent = "Unterstützungssitzung mit Yoka";
  toggleBtn.textContent = "Kamera ein/ausblenden";
}

startBtn.onclick = async () => {
  if (!hasStartedOnce) {
    hasStartedOnce = true;
    nextBtn.disabled = false;
  }

  if (!isRecording) {
    startBtn.classList.add("recording");
    startBtn.textContent = getLocalizedText("pause");
    await startRecording();
    await sendInitialMessage();
    isRecording = true;
  } else {
    startBtn.classList.remove("recording");
    startBtn.textContent = getLocalizedText("resume");
    stopAudioRecording();
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.pause();
    }
    if (recognition) recognition.stop();
    isRecording = false;
  }
};

async function startRecording() {
  if (mediaRecorder && mediaRecorder.state === "paused") {
    mediaRecorder.resume();
    listenToUser();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    videoEl.srcObject = stream;
    videoEl.style.display = "block";
    mediaRecorder = new MediaRecorder(stream);
    recordedChunks = [];

    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = uploadVideo;
    mediaRecorder.start();
  } catch (err) {
    alert("Please allow camera and microphone access.");
    console.error(err);
  }
}

function uploadVideo() {
  return new Promise((resolve, reject) => {
    const blob = new Blob(recordedChunks, { type: "video/webm" });
    const formData = new FormData();
    formData.append("video", blob, "user_video.webm");

    fetch("/upload_video", {
      method: "POST",
      body: formData
    }).then(res => {
      if (res.ok) {
        console.log("Video uploaded");
        resolve();
      } else {
        console.warn("Video upload failed");
        reject();
      }
    }).catch(err => {
      console.error("Video upload error:", err);
      reject();
    });
  });
}

function stopRecordingAndGo(url, showLoading = false) {
  async function clearAndRedirect() {
    try {
      const res = await fetch('/clear_files', { method: 'POST' });
      if (!res.ok) console.warn('Failed to clear files before exit.');
    } catch (e) {
      console.error('Error clearing files:', e);
    }
    if (showLoading) {
      document.getElementById("loading-screen").style.display = "flex";
    }
    window.location.href = url;
  }

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.onstop = () => clearAndRedirect();
    mediaRecorder.stop();
  } else {
    clearAndRedirect();
  }
}

toggleBtn.onclick = () => {
  videoEl.classList.toggle("hidden");
};

document.getElementById("exit-btn").onclick = () => stopRecordingAndGo("/home");

nextBtn.onclick = () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.onstop = () => {
      uploadVideo().then(() => {
        window.location.href = "/upload_eeg";
      }).catch(() => {
        window.location.href = "/upload_eeg";
      });
    };
    mediaRecorder.stop();
  } else {
    uploadVideo().then(() => {
      window.location.href = "/upload_eeg";
    }).catch(() => {
      window.location.href = "/upload_eeg";
    });
  }
};

async function sendInitialMessage() {
  try {
    await fetch("/generate_response", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "start recording" })
    });
    setTimeout(() => listenToUser(), 800);
  } catch (err) {
    console.error("Error during initial message:", err);
    setTimeout(() => listenToUser(), 800);
  }
}

let recognition;

async function listenToUser() {
  recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
  recognition.lang = language || "en";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    console.log("🎤 SpeechRecognition started");
    startAudioRecording();
  };

  recognition.onspeechend = () => {
    recognition.stop();
    stopAudioRecording();
  };

  recognition.onresult = async (event) => {
    const userText = event.results[0][0].transcript;
    console.log("👤 قال المستخدم:", userText);

    const response = await fetch("/generate_response", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: userText })
    });

    const data = await response.json();
    if (data.audio_url) {
      const audio = new Audio(data.audio_url);
      document.getElementById("chat-image").classList.add("speaking");
      audio.play();

      audio.onended = () => {
        document.getElementById("chat-image").classList.remove("speaking");
        setTimeout(() => listenToUser(), 800);
      };
    } else {
      setTimeout(() => listenToUser(), 800);
    }
  };

  recognition.onerror = (event) => {
    console.error("Error during recognition:", event.error);
    setTimeout(() => listenToUser(), 800);
  };

  recognition.start();
}

async function startAudioRecording() {
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioRecorder = new MediaRecorder(audioStream);
    audioChunks = [];

    audioRecorder.ondataavailable = e => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    audioRecorder.start();
  } catch (err) {
    console.error('Error starting audio recording:', err);
  }
}

function stopAudioRecording() {
  if (!audioRecorder) return;

  audioRecorder.onstop = async () => {
    const blob = new Blob(audioChunks, { type: 'audio/webm' });
    console.log('Uploading audio blob size:', blob.size);
    const formData = new FormData();
    formData.append('audio', blob, 'user_voice.webm');

    try {
      const response = await fetch('/process_audio', {
        method: 'POST',
        body: formData
      });
      if (!response.ok) console.error('Failed to upload audio', response.status);
      else console.log('Audio uploaded successfully');
    } catch (err) {
      console.error("Error uploading audio:", err);
    }

    if (audioStream) {
      audioStream.getTracks().forEach(track => track.stop());
      audioStream = null;
    }
  };

  audioRecorder.stop();
}
