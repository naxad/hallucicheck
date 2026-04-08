document.addEventListener("DOMContentLoaded", () => {
    const modeRadios = document.querySelectorAll('input[name="mode"]');
    const answerBlock = document.querySelector("#answerBlock");
    const submitBtn = document.querySelector("#submitBtn");

    function applyMode() {
        const selected = document.querySelector('input[name="mode"]:checked')?.value || "manual";

        if (selected === "chat") {
            if (answerBlock) answerBlock.style.display = "none";
            if (submitBtn) submitBtn.textContent = "Generate & Verify";
        } else {
            if (answerBlock) answerBlock.style.display = "block";
            if (submitBtn) submitBtn.textContent = "Verify";
        }
    }

    modeRadios.forEach(r => r.addEventListener("change", applyMode));
    applyMode();
});