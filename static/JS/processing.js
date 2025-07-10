// تحديد اللغة من العنصر <html lang="">
const lang = document.documentElement.lang || "en";

async function handleExit() {
  try {
    await fetch("/exit", { method: "GET" });
    window.location.href = "/home";
  } catch (error) {
    console.error("Error while exiting:", error);
    alert("Error while exiting");
  }
}

// نصوص حسب اللغة
const loadingTexts = {
  ar: "جارٍ تحليل البيانات...",
  en: "Analyzing your data...",
  fr: "Analyse des données...",
  de: "Daten werden analysiert..."
};

const errorTexts = {
  ar: "❌ حدث خطأ أثناء التحليل، من فضلك أعد المحاولة.",
  en: "❌ An error occurred during analysis. Please try again.",
  fr: "❌ Une erreur s'est produite lors de l'analyse. Veuillez réessayer.",
  de: "❌ Bei der Analyse ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut."
};



document.addEventListener("DOMContentLoaded", function () {
  document.getElementById("loading-text").textContent = loadingTexts[lang] || loadingTexts["en"];
  startAnalysis();
});


async function startAnalysis() {
  try {
    const response = await fetch("/analyze", {
      method: "POST"
    });

    const html = await response.text();

    document.open();
    document.write(html);
    document.close();

  } catch (error) {
    console.error("Error during analysis:", error);
    const loadingText = document.getElementById("loading-text");
    if (loadingText) {
      loadingText.textContent = errorTexts[lang] || errorTexts["en"];
    }
  }
}
