const detailedEl = document.getElementById("detailed-results");
const scoreValue = document.getElementById("score-value");
const descriptionEl = document.getElementById("score-description");
const homeBtn = document.getElementById("home-btn");

scoreValue.textContent = `${score}%`;


let message = "";
if (language === "ar") {
  if (score < 30) message = "مستوى الاكتئاب منخفض. استمر في العناية بنفسك.";
  else if (score < 70) message = "درجة اكتئاب متوسطة. يُفضل المتابعة مع مختص.";
  else message = "درجة الاكتئاب مرتفعة. من المهم طلب المساعدة.";
  homeBtn.textContent = "الصفحة الرئيسية";
} else if (language === "fr") {
  if (score < 30) message = "Niveau de dépression faible. Continuez à prendre soin de vous.";
  else if (score < 70) message = "Dépression modérée. Il est conseillé de consulter un spécialiste.";
  else message = "Niveau de dépression élevé. Demandez de l’aide si nécessaire.";
  homeBtn.textContent = "Accueil";
} else if (language === "de") {
  if (score < 30) message = "Niedrige Depressionswerte. Weiter so!";
  else if (score < 70) message = "Mittlere Depression. Bitte ziehen Sie eine Beratung in Betracht.";
  else message = "Hoher Depressionswert. Suchen Sie professionelle Hilfe.";
  homeBtn.textContent = "Startseite";
} else {
  if (score < 30) message = "Low depression level. Keep taking care of yourself.";
  else if (score < 70) message = "Moderate depression score. Consider consulting a professional.";
  else message = "High depression level. Please seek help.";
  homeBtn.textContent = "Home";
}

descriptionEl.textContent = message;

function translate(key, lang) {
  const labels = {
    audio: { ar: "الصوت", fr: "Audio", de: "Audio", en: "Audio" },
    video: { ar: "الفيديو", fr: "Vidéo", de: "Video", en: "Video" },
    text: { ar: "النص", fr: "Texte", de: "Text", en: "Text" },
    eeg: { ar: "الرسم المخي", fr: "EEG", de: "EEG", en: "EEG" },
    final: { ar: "الناتج النهائي", fr: "Score final", de: "Endwert", en: "Final Result" }
  };
  return labels[key] && labels[key][lang] || key;
}

function getNotProvidedText(key) {
  const notProvided = {
    ar: "لم يتم توفير الملف",
    fr: "Non fourni",
    de: "Nicht bereitgestellt",
    en: "Not provided"
  };
  return `<span class="not-provided">${notProvided[language] || notProvided["en"]}</span>`;
}

function goHome() {
  window.location.href = "/exit";
}
