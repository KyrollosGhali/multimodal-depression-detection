const language = localStorage.getItem("selected_language") || "en";

const questionTitle = document.getElementById("question-title");
const uploadLabel = document.getElementById("upload-label");
const yesBtn = document.getElementById("yes-btn");
const noBtn = document.getElementById("no-btn");
const resultBtn = document.getElementById("result-btn");
const uploadSection = document.getElementById("upload-section");
const eegFileInput = document.getElementById("eeg-file");
const form = document.getElementById("upload-form");
const exitBtn = document.getElementById("exit-btn");

if (language === "ar") {
  questionTitle.textContent = "هل لديك ملف رسم مخ؟";
  yesBtn.textContent = "نعم";
  noBtn.textContent = "لا";
  resultBtn.textContent = "عرض النتيجة";
  uploadLabel.textContent = "من فضلك اختر ملف رسم المخ (.edf أو .bdf)";
  exitBtn.textContent = "خروج 🚪";
} else if (language === "fr") {
  questionTitle.textContent = "Avez-vous un fichier EEG ?";
  yesBtn.textContent = "Oui";
  noBtn.textContent = "Non";
  resultBtn.textContent = "Afficher le résultat";
  uploadLabel.textContent = "Veuillez sélectionner votre fichier EEG (.edf ou .bdf)";
  exitBtn.textContent = "Sortie 🚪";
} else if (language === "de") {
  questionTitle.textContent = "Haben Sie eine EEG-Datei?";
  yesBtn.textContent = "Ja";
  noBtn.textContent = "Nein";
  resultBtn.textContent = "Ergebnis anzeigen";
  uploadLabel.textContent = "Bitte wählen Sie Ihre EEG-Datei (.edf oder .bdf)";
  exitBtn.textContent = "Ausgang 🚪";
} else {
  exitBtn.textContent = "Exit 🚪";
}

yesBtn.onclick = () => {
  uploadSection.classList.remove("hidden");
  resultBtn.classList.remove("hidden");
  eegFileInput.setAttribute("required", "required");
};

noBtn.onclick = () => {
  uploadSection.classList.add("hidden");
  resultBtn.classList.remove("hidden");
  eegFileInput.removeAttribute("required");
};

exitBtn.onclick = async () => {
  try {
    await fetch("/exit", { method: "GET" });
    window.location.href = "/home";
  } catch (error) {
    console.error("Error during exit:", error);
    alert("Error during exit");
  }
};

