(function () {
    "use strict";

    const MODEL_LABELS = {
        jay_cnn:           "Tumour Locator CNN",
        mark_cnn:          "Periocular Analysis System CNN",
        aparna_vit:        "Periocular Transformer - ViT (baseline)",
        aparna_vit_masked: "Periocular Transformer - ViT (iris-masked)",
    };

    // Human-readable labels for landmark measurements
    const LANDMARK_ITEMS = [
        { key: "palpebral_fissure_height", label: "Palpebral Fissure Height", unit: "px"  },
        { key: "palpebral_fissure_length", label: "Palpebral Fissure Length", unit: "px"  },
        { key: "aspect_ratio",             label: "Fissure Aspect Ratio",     unit: ""    },
        { key: "iris_radius",              label: "Iris Radius",              unit: "px"  },
        { key: "lesion_width",             label: "Lesion Width",             unit: "px"  },
        { key: "lesion_height",            label: "Lesion Height",            unit: "px"  },
        { key: "lesion_to_pupil_dist",     label: "Lesion-to-Pupil Distance", unit: "px"  },
    ];

    const API_PREDICT    = "/predict";
    const API_LANDMARKS  = "/analyse_landmarks";

    // ── State ─────────────────────────────────────────────────────────────────────
    let activePrototype = null;
    let selectedFiles   = [];

    // ── DOM refs ──────────────────────────────────────────────────────────────────
    const landingSection          = document.getElementById("landingSection");
    const protoABtn               = document.getElementById("protoABtn");
    const protoBBtn               = document.getElementById("protoBBtn");

    const uploadSection           = document.getElementById("uploadSection");
    const backToLandingFromUpload = document.getElementById("backToLandingFromUpload");
    const activeProtoLabel        = document.getElementById("activeProtoLabel");
    const uploadZone              = document.getElementById("uploadZone");
    const fileInput               = document.getElementById("fileInput");
    const previewContainer        = document.getElementById("previewContainer");
    const multiPreviewGrid        = document.getElementById("multiPreviewGrid");
    const fileCountText           = document.getElementById("fileCountText");
    const analyseBtn              = document.getElementById("analyseBtn");

    const resultsGallerySection   = document.getElementById("resultsGallerySection");
    const resultsGrid             = document.getElementById("resultsGrid");
    const metricsSummary          = document.getElementById("metricsSummary");
    const batchStatusText         = document.getElementById("batchStatusText");
    const galleryNewBtn           = document.getElementById("galleryNewBtn");
    const backToLandingBtn        = document.getElementById("backToLandingBtn");

    const errorToast   = document.getElementById("errorToast");
    const errorMessage = document.getElementById("errorMessage");
    const bgParticles  = document.getElementById("bgParticles");

    // ── Background particles ──────────────────────────────────────────────────────
    (function createParticles() {
        for (let i = 0; i < 25; i++) {
            const p    = document.createElement("div");
            p.classList.add("particle");
            const size = Math.random() * 3 + 1;
            const hue  = Math.random() > 0.5 ? "160" : "220";
            p.style.cssText = `
                width:${size}px;height:${size}px;
                left:${Math.random()*100}%;
                background:hsl(${hue},70%,70%);
                animation-duration:${Math.random()*15+12}s;
                animation-delay:${Math.random()*15}s;
            `;
            bgParticles.appendChild(p);
        }
    })();

    // ── Navigation ────────────────────────────────────────────────────────────────
    function showLanding() {
        landingSection.style.display = "";
        uploadSection.style.display  = "none";
        resultsGallerySection.classList.remove("visible");
        metricsSummary.innerHTML = "";
        activePrototype = null;
        resetUpload();
    }

    function showUpload(proto) {
        activePrototype = proto;
        landingSection.style.display = "none";
        uploadSection.style.display  = "";
        resultsGallerySection.classList.remove("visible");
        activeProtoLabel.textContent = proto === "A"
            ? "Prototype A — Dual CNN"
            : "Prototype B — Periocular Transformer";
        activeProtoLabel.dataset.proto = proto.toLowerCase();
        resetUpload();
    }

    function showResults() {
        uploadSection.style.display = "none";
        resultsGallerySection.classList.add("visible");
    }

    protoABtn.addEventListener("click", () => showUpload("A"));
    protoBBtn.addEventListener("click", () => showUpload("B"));
    backToLandingFromUpload.addEventListener("click", showLanding);
    backToLandingBtn.addEventListener("click", showLanding);
    galleryNewBtn.addEventListener("click", () => {
        resultsGallerySection.classList.remove("visible");
        metricsSummary.innerHTML = "";
        uploadSection.style.display = "";
        resetUpload();
    });

    // ── File handling ─────────────────────────────────────────────────────────────
    uploadZone.addEventListener("click", () => fileInput.click());
    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("drag-over");
    });
    uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length) handleFiles(Array.from(e.dataTransfer.files));
    });
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) handleFiles(Array.from(fileInput.files));
    });

    function handleFiles(files) {
        selectedFiles = [];
        multiPreviewGrid.innerHTML = "";
        const validTypes = ["image/jpeg","image/png","image/bmp","image/tiff","image/webp"];
        files.forEach(file => {
            if (!validTypes.includes(file.type) || file.size > 10*1024*1024) return;
            selectedFiles.push(file);
            const reader = new FileReader();
            reader.onload = (e) => {
                const item = document.createElement("div");
                item.classList.add("preview-item");
                item.innerHTML = `<img src="${e.target.result}" alt="${file.name}">`;
                multiPreviewGrid.appendChild(item);
            };
            reader.readAsDataURL(file);
        });
        if (selectedFiles.length > 0) {
            previewContainer.classList.add("visible");
            fileCountText.textContent = `${selectedFiles.length} image(s) selected`;
            uploadZone.style.display  = "none";
            analyseBtn.disabled       = false;
        } else {
            showError("No valid images selected. Please upload JPG or PNG files under 10 MB.");
        }
    }

    function resetUpload() {
        selectedFiles = [];
        fileInput.value             = "";
        multiPreviewGrid.innerHTML  = "";
        previewContainer.classList.remove("visible");
        fileCountText.textContent   = "";
        uploadZone.style.display    = "";
        analyseBtn.disabled         = true;
        analyseBtn.classList.remove("loading");
    }

    // ── Analyse button ────────────────────────────────────────────────────────────
    analyseBtn.addEventListener("click", async () => {
        if (!selectedFiles.length || !activePrototype) return;
        analyseBtn.disabled = true;
        analyseBtn.classList.add("loading");
        metricsSummary.innerHTML = "";
        showResults();
        if (activePrototype === "A") {
            await runPrototypeA();
        } else {
            await runPrototypeB();
        }
        analyseBtn.classList.remove("loading");
        analyseBtn.disabled = false;
    });

    // ── Prototype A: Jay + Mark in parallel ───────────────────────────────────────
    async function runPrototypeA() {
        const total     = selectedFiles.length;
        const jayPreds  = [];
        const markPreds = [];
        resultsGrid.innerHTML       = "";
        batchStatusText.textContent = "Running both CNN models in parallel…";

        for (let i = 0; i < total; i++) {
            const file = selectedFiles[i];
            const gt   = inferGroundTruth(file.name);
            batchStatusText.textContent =
                `Analysing image ${i+1} of ${total} with Tumour Locator + Periocular Analysis CNNs…`;

            const jayForm  = buildForm(file, "jay_cnn");
            const markForm = buildForm(file, "mark_cnn");

            try {
                const [jayData, markData] = await Promise.all([
                    fetchPredict(jayForm),
                    fetchPredict(markForm),
                ]);
                appendDualCard(file.name, jayData, markData);
                jayPreds.push({  predicted: jayData.prediction,  groundTruth: gt });
                markPreds.push({ predicted: markData.prediction, groundTruth: gt });
            } catch (err) {
                console.error(err);
                appendErrorCard(file.name, err.message);
            }
        }

        batchStatusText.textContent =
            `✓ Done — ${total} image${total!==1?"s":""} analysed with Prototype A.`;
        renderMetricsSummary([
            { modelType: "jay_cnn",  predictions: jayPreds  },
            { modelType: "mark_cnn", predictions: markPreds },
        ]);
    }

    // ── Prototype B: Landmark analysis + baseline ViT + iris-masked ViT ──────────
    async function runPrototypeB() {
        const total       = selectedFiles.length;
        const vitPreds    = [];
        const maskedPreds = [];
        resultsGrid.innerHTML       = "";
        batchStatusText.textContent = "Running geometric landmark analysis + Periocular Transformer (baseline & iris-masked)…";

        for (let i = 0; i < total; i++) {
            const file = selectedFiles[i];
            const gt   = inferGroundTruth(file.name);
            batchStatusText.textContent =
                `Analysing image ${i+1} of ${total} — landmarks + ViT (baseline & iris-masked)…`;

            const vitForm    = buildForm(file, "aparna_vit");
            const maskedForm = buildForm(file, "aparna_vit_masked");

            // Run landmark analysis and both ViT variants simultaneously; use
            // allSettled so one failure doesn't block the others.
            const [lmResult, vitResult, maskedResult] = await Promise.allSettled([
                fetchLandmarks(file),
                fetchPredict(vitForm),
                fetchPredict(maskedForm),
            ]);

            const lmData     = lmResult.status     === "fulfilled" ? lmResult.value     : null;
            const vitData    = vitResult.status    === "fulfilled" ? vitResult.value    : null;
            const maskedData = maskedResult.status === "fulfilled" ? maskedResult.value : null;

            if (vitData || maskedData) {
                appendProtoBCard(file.name, lmData, vitData, maskedData);
                if (vitData)    vitPreds.push({    predicted: vitData.prediction,    groundTruth: gt });
                if (maskedData) maskedPreds.push({ predicted: maskedData.prediction, groundTruth: gt });
            } else {
                appendErrorCard(file.name, vitResult.reason?.message || maskedResult.reason?.message || "Analysis failed");
            }
        }

        batchStatusText.textContent =
            `✓ Done — ${total} image${total!==1?"s":""} analysed with Prototype B.`;
        renderMetricsSummary([
            { modelType: "aparna_vit",        predictions: vitPreds    },
            { modelType: "aparna_vit_masked", predictions: maskedPreds },
        ]);
    }

    // ── Fetch helpers ─────────────────────────────────────────────────────────────
    function buildForm(file, modelType) {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("model_type", modelType);
        return fd;
    }

    async function fetchPredict(formData) {
        const res = await fetch(API_PREDICT, { method: "POST", body: formData });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: `Server error ${res.status}` }));
            throw new Error(err.detail || `Server error ${res.status}`);
        }
        return res.json();
    }

    async function fetchLandmarks(file) {
        const fd = new FormData();
        fd.append("file", file);
        const res = await fetch(API_LANDMARKS, { method: "POST", body: fd });
        if (!res.ok) throw new Error(`Landmark analysis failed (${res.status})`);
        return res.json();
    }

    // ── Ground truth inference ────────────────────────────────────────────────────
    // Dataset naming convention: B*.jpg = Benign, M*.jpg = Malignant
    function inferGroundTruth(filename) {
        const name = filename.replace(/.*[\\/]/, "");
        if (/^m/i.test(name)) return "Malignant";
        if (/^b/i.test(name)) return "Benign";
        return null;
    }

    // ── Metrics computation ───────────────────────────────────────────────────────
    // Positive class = Malignant
    function computeMetrics(predictions) {
        let tp = 0, fp = 0, tn = 0, fn = 0, unlab = 0;
        for (const p of predictions) {
            if (!p.groundTruth) { unlab++; continue; }
            const predMal = p.predicted   === "Malignant";
            const gtMal   = p.groundTruth === "Malignant";
            if      ( predMal &&  gtMal) tp++;
            else if ( predMal && !gtMal) fp++;
            else if (!predMal &&  gtMal) fn++;
            else                          tn++;
        }
        const labeled = tp + fp + tn + fn;
        return {
            tp, fp, tn, fn, labeled, unlab,
            accuracy:  labeled > 0       ? (tp+tn)/labeled  : null,
            precision: (tp+fp) > 0       ? tp/(tp+fp)       : null,
            recall:    (tp+fn) > 0       ? tp/(tp+fn)       : null,
        };
    }

    // ── Metrics summary rendering ─────────────────────────────────────────────────
    function renderMetricsSummary(modelPredictions) {
        const hasLabels = modelPredictions.some(mp =>
            mp.predictions.some(p => p.groundTruth !== null)
        );

        const section = document.createElement("div");
        section.classList.add("metrics-summary-section");

        const header = `
            <div class="metrics-summary-header">
                <h3 class="metrics-summary-title">Performance Summary</h3>
                <p class="metrics-summary-note">
                    Ground truth inferred from filename &mdash;
                    <code>B*</code>&thinsp;=&thinsp;Benign &middot;
                    <code>M*</code>&thinsp;=&thinsp;Malignant &middot;
                    Positive class&thinsp;=&thinsp;Malignant
                </p>
            </div>`;

        if (!hasLabels) {
            section.innerHTML = header + `
                <div class="metrics-no-labels">
                    <svg viewBox="0 0 20 20" fill="currentColor" width="20" height="20">
                        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
                    </svg>
                    <p>No ground truth labels detected in filenames. Rename images as
                    <code>B001.jpg</code>&nbsp;(Benign) or <code>M001.jpg</code>&nbsp;(Malignant)
                    to compute accuracy, precision, and recall.</p>
                </div>`;
        } else {
            const cardsHtml = modelPredictions
                .map(mp => buildMetricsCard(MODEL_LABELS[mp.modelType], computeMetrics(mp.predictions)))
                .join("");
            section.innerHTML = header + `<div class="metrics-cards-grid">${cardsHtml}</div>`;
        }

        metricsSummary.appendChild(section);
    }

    function buildMetricsCard(modelName, m) {
        const fmt = (v) => v !== null ? `${(v*100).toFixed(1)}%` : "—";

        const countNote = m.labeled === 0
            ? `<span class="metrics-count-note metrics-count-none">No labelled images</span>`
            : `<span class="metrics-count-note">${m.labeled} labelled image${m.labeled!==1?"s":""}${m.unlab>0?` &middot; ${m.unlab} unlabelled`:""}</span>`;

        if (m.labeled === 0) {
            return `
                <div class="metrics-card">
                    <div class="metrics-card-header">
                        <h4 class="metrics-card-model">${modelName}</h4>${countNote}
                    </div>
                    <p class="metrics-empty">Upload images with B* or M* filenames to see metrics.</p>
                </div>`;
        }

        return `
            <div class="metrics-card">
                <div class="metrics-card-header">
                    <h4 class="metrics-card-model">${modelName}</h4>${countNote}
                </div>
                <div class="conf-matrix">
                    <p class="conf-matrix-title">Confusion Matrix</p>
                    <div class="conf-matrix-grid">
                        <div class="conf-corner"></div>
                        <div class="conf-col-header">Pred:&nbsp;Malignant</div>
                        <div class="conf-col-header">Pred:&nbsp;Benign</div>
                        <div class="conf-row-header">Act:&nbsp;Malignant</div>
                        <div class="conf-cell conf-tp"><span class="conf-num">${m.tp}</span><span class="conf-abbr">TP</span></div>
                        <div class="conf-cell conf-fn"><span class="conf-num">${m.fn}</span><span class="conf-abbr">FN</span></div>
                        <div class="conf-row-header">Act:&nbsp;Benign</div>
                        <div class="conf-cell conf-fp"><span class="conf-num">${m.fp}</span><span class="conf-abbr">FP</span></div>
                        <div class="conf-cell conf-tn"><span class="conf-num">${m.tn}</span><span class="conf-abbr">TN</span></div>
                    </div>
                </div>
                <div class="formula-list">
                    <div class="formula-row">
                        <span class="formula-name">Accuracy</span>
                        <div class="formula-body">
                            <span class="formula-def">= (TP + TN) / (TP + TN + FP + FN)</span>
                            <span class="formula-sub">= (${m.tp} + ${m.tn}) / (${m.tp} + ${m.tn} + ${m.fp} + ${m.fn})</span>
                        </div>
                        <span class="formula-result">${fmt(m.accuracy)}</span>
                    </div>
                    <div class="formula-row">
                        <span class="formula-name">Precision</span>
                        <div class="formula-body">
                            <span class="formula-def">= TP / (TP + FP)</span>
                            <span class="formula-sub">= ${m.tp} / (${m.tp} + ${m.fp})</span>
                        </div>
                        <span class="formula-result">${fmt(m.precision)}</span>
                    </div>
                    <div class="formula-row">
                        <span class="formula-name">Recall</span>
                        <div class="formula-body">
                            <span class="formula-def">= TP / (TP + FN)</span>
                            <span class="formula-sub">= ${m.tp} / (${m.tp} + ${m.fn})</span>
                        </div>
                        <span class="formula-result">${fmt(m.recall)}</span>
                    </div>
                </div>
            </div>`;
    }

    // ── Prototype B card (landmark step + baseline ViT step + masked ViT step) ────
    function appendProtoBCard(filename, lmData, vitData, maskedData) {
        const card = document.createElement("div");
        card.classList.add("batch-card--protob");
        card.innerHTML = `
            <div class="protob-header">
                <h4 class="case-label">Case</h4>
                <p class="case-filename">${filename}</p>
            </div>
            ${buildLandmarkStep(lmData)}
            ${vitData    ? buildViTStep(vitData)             : ""}
            ${maskedData ? buildMaskedViTStep(maskedData)     : ""}
        `;
        resultsGrid.appendChild(card);
    }

    function buildLandmarkStep(lmData) {
        const stepBadge = `
            <div class="step-label">
                <span class="step-num">1</span>
                Geometric Landmark Analysis
            </div>`;

        if (!lmData) {
            return `<div class="analysis-step">
                ${stepBadge}
                <p class="landmark-failed">Landmark detection could not be completed for this image.</p>
            </div>`;
        }

        const rows = LANDMARK_ITEMS
            .filter(item => lmData.measurements && lmData.measurements[item.key] != null)
            .map(item => `
                <tr>
                    <td class="lm-key">${item.label}</td>
                    <td class="lm-val">${lmData.measurements[item.key]}${item.unit ? " " + item.unit : ""}</td>
                </tr>`).join("");

        const table = rows
            ? `<table class="landmark-table"><tbody>${rows}</tbody></table>`
            : `<p class="landmark-no-detect">No anatomical landmarks detected in this image.</p>`;

        const img = lmData.annotated_image
            ? `<div class="landmark-img-col">
                   <div class="heatmap-img-wrapper">
                       <img src="data:image/jpeg;base64,${lmData.annotated_image}" alt="Landmark overlay">
                   </div>
                   <span class="heatmap-label">Landmark Overlay</span>
               </div>`
            : "";

        return `
            <div class="analysis-step">
                ${stepBadge}
                <div class="landmark-content">
                    ${img}
                    <div class="landmark-metrics-col">${table}</div>
                </div>
            </div>`;
    }

    function buildViTStep(vitData) {
        const isMal = vitData.prediction === "Malignant";
        const color = isMal ? "var(--malignant-color)" : "var(--benign-color)";
        const cols  = vitData.heatmap ? 2 : 1;

        return `
            <div class="analysis-step">
                <div class="step-label step-label--vit">
                    <span class="step-num step-num--vit">2</span>
                    Periocular Transformer &mdash; ViT (baseline, no masking)
                </div>
                <div class="vit-content">
                    <div class="vit-pred-col">
                        <div class="prediction-row" style="margin-top:0;">
                            <span class="prediction-model-name">${MODEL_LABELS.aparna_vit}</span>
                            <div class="prediction-values">
                                <span class="prediction-label" style="color:${color};">${vitData.prediction}</span>
                                <span class="prediction-confidence">${vitData.confidence}% confidence</span>
                            </div>
                        </div>
                    </div>
                    <div class="vit-imgs-col">
                        <div class="heatmap-grid" style="grid-template-columns:repeat(${cols},1fr);gap:12px;">
                            ${vitData.original_image ? buildHeatmapItem(vitData.original_image, "Original Image")   : ""}
                            ${vitData.heatmap        ? buildHeatmapItem(vitData.heatmap,        "Attention / GMAR") : ""}
                        </div>
                    </div>
                </div>
            </div>`;
    }

    // ── Prototype B step 3: iris-masked ViT (ViTbackend, improved model) ──────
    function buildMaskedViTStep(maskedData) {
        const isMal = maskedData.prediction === "Malignant";
        const color = isMal ? "var(--malignant-color)" : "var(--benign-color)";
        const imgs  = [
            maskedData.masked_image ? { b64: maskedData.masked_image, label: "Iris-Masked Input" } : null,
            maskedData.heatmap      ? { b64: maskedData.heatmap,      label: "Attention / GMAR"   } : null,
        ].filter(Boolean);

        return `
            <div class="analysis-step">
                <div class="step-label step-label--vit-masked">
                    <span class="step-num step-num--vit-masked">3</span>
                    Periocular Transformer &mdash; ViT (iris-masked, improved)
                </div>
                <div class="vit-content">
                    <div class="vit-pred-col">
                        <div class="prediction-row" style="margin-top:0;">
                            <span class="prediction-model-name">${MODEL_LABELS.aparna_vit_masked}</span>
                            <div class="prediction-values">
                                <span class="prediction-label" style="color:${color};">${maskedData.prediction}</span>
                                <span class="prediction-confidence">${maskedData.confidence}% confidence</span>
                            </div>
                        </div>
                    </div>
                    <div class="vit-imgs-col">
                        <div class="heatmap-grid" style="grid-template-columns:repeat(${imgs.length || 1},1fr);gap:12px;">
                            ${imgs.map(img => buildHeatmapItem(img.b64, img.label)).join("")}
                        </div>
                    </div>
                </div>
            </div>`;
    }

    // ── Prototype A dual card ─────────────────────────────────────────────────────
    function appendDualCard(filename, jayData, markData) {
        const card = document.createElement("div");
        card.classList.add("batch-card", "batch-card--dual");
        card.innerHTML = `
            <div class="batch-card-info">
                <h4 class="case-label">Case</h4>
                <p class="case-filename">${filename}</p>
                ${buildPredictionRow(jayData,  "jay_cnn")}
                ${buildPredictionRow(markData, "mark_cnn")}
            </div>
            <div class="batch-card-visuals">
                ${jayData.original_image ? buildVisualCol(jayData.original_image, "Original Image")                : ""}
                ${jayData.heatmap        ? buildVisualCol(jayData.heatmap,        "Tumour Locator / Grad-CAM")      : ""}
                ${markData.heatmap       ? buildVisualCol(markData.heatmap,       "Periocular Analysis / Grad-CAM") : ""}
            </div>`;
        resultsGrid.appendChild(card);
    }

    function buildPredictionRow(data, modelType) {
        const isMal = data.prediction === "Malignant";
        const color = isMal ? "var(--malignant-color)" : "var(--benign-color)";
        return `
            <div class="prediction-row">
                <span class="prediction-model-name">${MODEL_LABELS[modelType]}</span>
                <div class="prediction-values">
                    <span class="prediction-label" style="color:${color};">${data.prediction}</span>
                    <span class="prediction-confidence">${data.confidence}% confidence</span>
                </div>
            </div>`;
    }

    function buildVisualCol(b64, label) {
        return `
            <div class="visual-col">
                <div class="heatmap-img-wrapper">
                    <img src="data:image/jpeg;base64,${b64}" alt="${label}">
                </div>
                <span class="heatmap-label">${label}</span>
            </div>`;
    }

    function buildHeatmapItem(b64, label) {
        return `
            <div class="heatmap-item">
                <div class="heatmap-img-wrapper">
                    <img src="data:image/jpeg;base64,${b64}" alt="${label}">
                </div>
                <span class="heatmap-label">${label}</span>
            </div>`;
    }

    function appendErrorCard(filename, error) {
        const card = document.createElement("div");
        card.classList.add("batch-card", "batch-card--error");
        card.innerHTML = `
            <div class="batch-card-info">
                <h4 class="case-label">Case</h4>
                <p class="case-filename">${filename}</p>
                <div class="error-label">Analysis Failed</div>
                <p class="error-detail">${error}</p>
            </div>`;
        resultsGrid.appendChild(card);
    }

    // ── Error toast ───────────────────────────────────────────────────────────────
    let errorTimeout;
    function showError(msg) {
        errorMessage.textContent = msg;
        errorToast.classList.add("show");
        clearTimeout(errorTimeout);
        errorTimeout = setTimeout(() => errorToast.classList.remove("show"), 5000);
    }

})();
