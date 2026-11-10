(function () {
    "use strict";

    const MAX_SIZE = 10 * 1024 * 1024;

    // Measurements returned by /api/landmarks (pixels in the 224x224 image).
    const LANDMARK_ITEMS = [
        { key: "palpebral_fissure_height", label: "Palpebral fissure height", unit: "px" },
        { key: "palpebral_fissure_length", label: "Palpebral fissure length", unit: "px" },
        { key: "aspect_ratio",             label: "Fissure aspect ratio",     unit: ""   },
        { key: "iris_radius",              label: "Iris radius",              unit: "px" },
        { key: "lesion_width",             label: "Lesion width",             unit: "px" },
        { key: "lesion_height",            label: "Lesion height",            unit: "px" },
        { key: "lesion_to_pupil_dist",     label: "Lesion-to-pupil distance", unit: "px" },
    ];
    const EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"];

    const $ = (id) => document.getElementById(id);
    const statusEl = $("status");
    const banner = $("banner");
    const uploadSection = $("uploadSection");
    const dropZone = $("dropZone");
    const fileInput = $("fileInput");
    const selection = $("selection");
    const thumbs = $("thumbs");
    const analyseBtn = $("analyseBtn");
    const resultsSection = $("resultsSection");
    const results = $("results");
    const progress = $("progress");
    const newBtn = $("newBtn");

    let files = [];
    let modelReady = false;

    // ── Model status ──────────────────────────────────────────────────────────
    fetch("/api/status")
        .then((r) => r.json())
        .then((s) => {
            modelReady = s.ready;
            statusEl.textContent = s.ready ? "Model ready" : "Model unavailable";
            statusEl.classList.add(s.ready ? "ok" : "bad");
            if (!s.ready) showBanner(s.error || "The model could not be loaded.");
            updateAnalyseButton();
        })
        .catch(() => {
            statusEl.textContent = "Server offline";
            statusEl.classList.add("bad");
            showBanner("Cannot reach the server. Is main.py running?");
        });

    function showBanner(msg) {
        banner.textContent = msg;
        banner.hidden = false;
    }

    // ── File selection ────────────────────────────────────────────────────────
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        addFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener("change", () => {
        addFiles(fileInput.files);
        fileInput.value = "";
    });

    function isAccepted(file) {
        const name = file.name.toLowerCase();
        return EXTENSIONS.some((ext) => name.endsWith(ext)) && file.size <= MAX_SIZE;
    }

    function addFiles(list) {
        const all = Array.from(list);
        const accepted = all.filter(isAccepted);
        if (accepted.length < all.length) {
            showBanner(`${all.length - accepted.length} file(s) skipped: only images up to 10 MB are accepted.`);
        }
        files = files.concat(accepted);
        renderSelection();
    }

    function renderSelection() {
        thumbs.replaceChildren(...files.map((file) => {
            const img = document.createElement("img");
            img.src = URL.createObjectURL(file);
            img.alt = file.name;
            img.title = file.name;
            img.onload = () => URL.revokeObjectURL(img.src);
            return img;
        }));
        selection.hidden = files.length === 0;
        updateAnalyseButton();
    }

    function updateAnalyseButton() {
        // Landmark analysis needs no model, so analysis is allowed even if the model failed to load.
        analyseBtn.disabled = files.length === 0;
        analyseBtn.textContent = files.length > 1 ? `Analyse ${files.length} images` : "Analyse";
    }

    $("clearBtn").addEventListener("click", reset);
    newBtn.addEventListener("click", reset);

    function reset() {
        files = [];
        renderSelection();
        results.replaceChildren();
        resultsSection.hidden = true;
        uploadSection.hidden = false;
        if (modelReady) banner.hidden = true;
    }

    // ── Analysis ──────────────────────────────────────────────────────────────
    analyseBtn.addEventListener("click", async () => {
        const batch = files.slice();
        uploadSection.hidden = true;
        resultsSection.hidden = false;
        newBtn.disabled = true;
        results.replaceChildren();

        for (let i = 0; i < batch.length; i++) {
            progress.textContent = `Analysing image ${i + 1} of ${batch.length}…`;
            const card = el("article", "card pending");
            card.append(el("h3", "filename", batch[i].name), el("p", "muted", "Analysing…"));
            results.append(card);
            // Classification and landmark analysis are independent; run both and
            // show whichever succeeds.
            const [pred, lm] = await Promise.allSettled([
                post("/api/predict", batch[i]),
                post("/api/landmarks", batch[i]),
            ]);
            fillCard(card, batch[i].name, pred, lm);
        }
        progress.textContent = `Done — ${batch.length} image${batch.length === 1 ? "" : "s"} analysed.`;
        newBtn.disabled = false;
    });

    async function post(url, file) {
        const form = new FormData();
        form.append("file", file);
        const res = await fetch(url, { method: "POST", body: form });
        const body = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(body.detail || `Server error ${res.status}`);
        return body;
    }

    function fillCard(card, filename, pred, lm) {
        card.className = "card";
        const summary = el("div", "summary");
        summary.append(el("h3", "filename", filename));

        if (pred.status === "fulfilled") {
            const r = pred.value;
            const malignant = r.prediction === "Malignant";
            summary.append(
                el("p", "prediction " + (malignant ? "malignant" : "benign"), r.prediction),
                el("p", "muted", `${r.confidence}% confidence · P(malignant) = ${r.probability_malignant.toFixed(3)}`),
            );
            if (r.iris_method === "skipped") {
                summary.append(el("p", "warn", "No iris detected — image analysed without masking."));
            }
            const images = el("div", "images");
            images.append(
                figure(r.original_image, "Original"),
                figure(r.masked_image, "Model input (CLAHE + iris mask)"),
                figure(r.heatmap, "GMAR attention"),
            );
            card.replaceChildren(summary, images);
        } else {
            summary.append(el("p", "error-text", `Classification failed: ${pred.reason.message}`));
            card.replaceChildren(summary, el("div"));
        }

        card.append(landmarkSection(lm));
    }

    function landmarkSection(lm) {
        const section = el("div", "landmarks");
        section.append(el("h4", "", "Geometric landmark analysis"));

        if (lm.status !== "fulfilled") {
            section.append(el("p", "error-text", `Landmark analysis failed: ${lm.reason.message}`));
            return section;
        }

        const m = lm.value.measurements || {};
        const rows = LANDMARK_ITEMS.filter((item) => m[item.key] != null).map((item) => {
            const tr = document.createElement("tr");
            tr.append(el("td", "", item.label), el("td", "value", `${m[item.key]}${item.unit ? " " + item.unit : ""}`));
            return tr;
        });

        let detail;
        if (rows.length) {
            detail = document.createElement("table");
            detail.append(...rows);
        } else {
            detail = el("p", "muted", "No anatomical landmarks detected in this image.");
        }

        const body = el("div", "landmarks-body");
        body.append(figure(lm.value.annotated_image, "Landmark overlay"), detail);
        section.append(body);
        return section;
    }

    function figure(b64, caption) {
        const fig = el("figure");
        const img = document.createElement("img");
        img.src = `data:image/jpeg;base64,${b64}`;
        img.alt = caption;
        fig.append(img, el("figcaption", "", caption));
        return fig;
    }

    function el(tag, className = "", text = "") {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text) node.textContent = text;
        return node;
    }
})();
