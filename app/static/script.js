(function () {
    "use strict";

    const MAX_SIZE = 10 * 1024 * 1024;
    const EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"];
    const COMMENTS_MAX = 500;

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
    const startOverBtn = $("startOverBtn");
    const saveBar = $("saveBar");
    const saveBtn = $("saveBtn");
    const nextBtn = $("nextBtn");
    const saveMessage = $("saveMessage");

    let files = [];
    let modelReady = false;
    // One entry per analysed image that can be evaluated:
    // { file, result, form: {...inputs}, annotator, card, recordId }
    let entries = [];
    let objectUrls = [];

    // ── Model status ──────────────────────────────────────────────────────────
    fetch("/api/status")
        .then((r) => r.json())
        .then((s) => {
            modelReady = s.ready;
            statusEl.textContent = s.ready ? "Model ready" : "Model unavailable";
            statusEl.classList.add(s.ready ? "ok" : "bad");
            if (!s.ready) showBanner(s.error || "The model could not be loaded.");
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
        analyseBtn.disabled = files.length === 0;
        analyseBtn.textContent = files.length > 1 ? `Analyse ${files.length} images` : "Analyse";
    }

    $("clearBtn").addEventListener("click", reset);
    nextBtn.addEventListener("click", reset);
    startOverBtn.addEventListener("click", () => {
        if (hasUnsavedWork() && !confirm("Discard the results and any unsaved evaluations?")) return;
        reset();
    });

    // FR-11: back to an empty upload page.
    function reset() {
        files = [];
        entries = [];
        objectUrls.forEach(URL.revokeObjectURL);
        objectUrls = [];
        renderSelection();
        results.replaceChildren();
        resultsSection.hidden = true;
        saveBar.hidden = true;
        uploadSection.hidden = false;
        if (modelReady) banner.hidden = true;
        window.scrollTo(0, 0);
    }

    function hasUnsavedWork() {
        return entries.some((e) => !e.recordId);
    }

    window.addEventListener("beforeunload", (e) => {
        if (hasUnsavedWork()) e.preventDefault();
    });

    // ── Analysis ──────────────────────────────────────────────────────────────
    analyseBtn.addEventListener("click", async () => {
        const batch = files.slice();
        uploadSection.hidden = true;
        resultsSection.hidden = false;
        saveBar.hidden = true;
        startOverBtn.disabled = true;
        results.replaceChildren();
        entries = [];

        for (let i = 0; i < batch.length; i++) {
            progress.textContent = `Analysing image ${i + 1} of ${batch.length}…`;
            const card = el("article", "card");
            card.append(el("h3", "filename", batch[i].name), el("p", "muted", "Analysing…"));
            results.append(card);
            // Classification and landmark analysis are independent; run both and
            // show whichever succeeds.
            const [pred, lm] = await Promise.allSettled([
                post("/api/predict", batch[i]),
                post("/api/landmarks", batch[i]),
            ]);
            buildCard(card, batch[i], pred, lm);
        }

        progress.textContent = `${batch.length} image${batch.length === 1 ? "" : "s"} analysed.`;
        startOverBtn.disabled = false;
        if (entries.length) {
            saveBar.hidden = false;
            saveBtn.hidden = false;
            saveBtn.disabled = false;
            saveBtn.textContent = entries.length > 1 ? `Save Evaluation (${entries.length} images)` : "Save Evaluation";
            nextBtn.hidden = true;
            setSaveMessage("", "");
        }
    });

    async function post(url, file) {
        const form = new FormData();
        form.append("file", file);
        const res = await fetch(url, { method: "POST", body: form });
        const body = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(body.detail || `Server error ${res.status}`);
        return body;
    }

    // ── Result card: model result, landmarks, then the three evaluation sections
    function buildCard(card, file, pred, lm) {
        card.replaceChildren();
        const head = el("div", "card-head");
        head.append(el("h3", "filename", file.name));
        const savedTag = el("span", "saved-tag");
        savedTag.hidden = true;
        head.append(savedTag);
        card.append(head);

        card.append(resultSection(pred), landmarkSection(lm));

        if (pred.status !== "fulfilled") {
            card.append(el("p", "muted", "This image can't be evaluated because the model did not produce a result."));
            return;
        }

        const entry = { file, result: pred.value, card, savedTag, recordId: null };
        card.append(evaluationSection(entry), annotationSection(entry), medicalSection(entry));
        entries.push(entry);
    }

    function resultSection(pred) {
        const section = el("section", "block");
        section.append(el("h4", "", "Iris-Masked Periocular Transformer"));

        if (pred.status !== "fulfilled") {
            section.append(el("p", "error-text", `Classification failed: ${pred.reason.message}`));
            return section;
        }
        const r = pred.value;
        const body = el("div", "result-body");
        const summary = el("div", "summary");
        summary.append(
            el("p", "prediction " + (r.prediction === "Malignant" ? "malignant" : "benign"), r.prediction),
            el("p", "muted", `${r.confidence}% confidence · P(malignant) = ${r.probability_malignant.toFixed(3)}`),
        );
        if (r.iris_method === "skipped") {
            summary.append(el("p", "warn", "No iris detected — image analysed without masking."));
        }
        const images = el("div", "images");
        images.append(
            figure(dataUrl(r.original_image), "Original"),
            figure(dataUrl(r.masked_image), "Model input (CLAHE + iris mask)"),
            figure(dataUrl(r.heatmap), "GMAR attention"),
        );
        body.append(summary, images);
        section.append(body);
        return section;
    }

    function landmarkSection(lm) {
        const section = el("section", "block");
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
        body.append(figure(dataUrl(lm.value.annotated_image), "Landmark overlay"), detail);
        section.append(body);
        return section;
    }

    // FR-05
    function evaluationSection(entry) {
        const section = el("section", "block form-block");
        section.append(el("h4", "", "Evaluation"), el("p", "form-intro", "To be completed by the Clinical User:"));

        const now = new Date();
        entry.evaluationDate = now;

        const name = input("text", { maxLength: 200, autocomplete: "name" });
        const when = input("datetime-local", { readOnly: true, value: localDateTimeValue(now) });
        const performance = select(["Good", "Poor", "Unknown"]);
        const comments = textarea(COMMENTS_MAX);

        section.append(
            field("Name of Clinical User", name, { required: true }),
            field("Date and Time", when, { hint: "Filled in automatically" }),
            field("Performance", performance, { required: true }),
            field("Comments", comments, { counter: COMMENTS_MAX }),
        );
        entry.form = { name, performance, comments };
        return section;
    }

    // FR-06 / FR-07
    function annotationSection(entry) {
        const section = el("section", "block form-block");
        section.append(
            el("h4", "", "Image Annotation"),
            el("p", "form-intro",
                "Click on the image below to indicate the region that should influence the model's decision " +
                "(this may include the lesion and any architectural distortion elsewhere)."),
        );
        const holder = el("div", "annotation-holder");
        const url = URL.createObjectURL(entry.file);
        objectUrls.push(url);
        entry.annotator = new Annotator(holder, url, dataUrl(entry.result.original_image));

        const clearBtn = el("button", "btn secondary small", "Clear annotations");
        clearBtn.type = "button";
        clearBtn.disabled = true;
        clearBtn.addEventListener("click", () => entry.annotator.clear());
        entry.annotator.onchange = () => { clearBtn.disabled = entry.annotator.lines.length === 0; };
        entry.clearAnnotationsBtn = clearBtn;

        const actions = el("div", "annotation-actions");
        actions.append(clearBtn);
        section.append(holder, actions);
        return section;
    }

    // FR-08 (all fields optional: the Consultant may not be present)
    function medicalSection(entry) {
        const section = el("section", "block form-block");
        section.append(el("h4", "", "Medical Assessment"), el("p", "form-intro", "To be completed by the Consultant:"));

        const consultant = input("text", { maxLength: 200 });
        const assessment = select(["Benign", "Malignant", "Don't know"]);
        const medComments = textarea(5000);

        section.append(
            field("Name of Consultant", consultant),
            field("Medical Assessment of Image", assessment),
            field("Medical Comments", medComments),
        );
        Object.assign(entry.form, { consultant, assessment, medComments });
        return section;
    }

    // ── Saving (FR-09, FR-10) ─────────────────────────────────────────────────
    saveBtn.addEventListener("click", async () => {
        const pending = entries.filter((e) => !e.recordId);
        const invalid = pending.filter((e) => !validate(e));
        if (invalid.length) {
            setSaveMessage(
                `Please complete the required fields (Name of Clinical User and Performance) for ${invalid.length} image${invalid.length === 1 ? "" : "s"}.`,
                "error",
            );
            invalid[0].card.querySelector(".invalid").focus();
            return;
        }

        saveBtn.disabled = true;
        startOverBtn.disabled = true;
        setSaveMessage("Saving…", "");

        const failures = [];
        for (const entry of pending) {
            setLocked(entry, true);
            try {
                const { record_id } = await saveEntry(entry);
                entry.recordId = record_id;
                entry.savedTag.textContent = `Saved · ${record_id}`;
                entry.savedTag.hidden = false;
            } catch (err) {
                setLocked(entry, false);
                failures.push(`${entry.file.name}: ${err.message}`);
            }
        }

        startOverBtn.disabled = false;
        const saved = entries.filter((e) => e.recordId);
        if (failures.length) {
            saveBtn.disabled = false;
            saveBtn.textContent = "Retry saving";
            setSaveMessage(
                `${failures.length} evaluation${failures.length === 1 ? "" : "s"} could not be saved. Your entries are kept — fix the problem and retry.\n` +
                failures.join("\n"),
                "error",
            );
            return;
        }

        saveBtn.hidden = true;
        nextBtn.hidden = false;
        setSaveMessage(
            `Evaluation${saved.length === 1 ? "" : "s"} saved. Record ID${saved.length === 1 ? "" : "s"}:\n` +
            saved.map((e) => `${e.recordId} — ${e.file.name}`).join("\n"),
            "ok",
        );
        nextBtn.focus();
    });

    function validate(entry) {
        const { name, performance } = entry.form;
        let ok = true;
        for (const control of [name, performance]) {
            const missing = !control.value.trim();
            control.classList.toggle("invalid", missing);
            if (missing) ok = false;
        }
        return ok;
    }

    async function saveEntry(entry) {
        const f = entry.form;
        const r = entry.result;
        const record = {
            clinical_user: {
                name: f.name.value.trim(),
                evaluation_datetime: isoWithOffset(entry.evaluationDate),
                performance: f.performance.value,
                comments: f.comments.value.trim(),
            },
            consultant: {
                name: f.consultant.value.trim(),
                medical_assessment: f.assessment.value || null,
                medical_comments: f.medComments.value.trim(),
            },
            ai_result: {
                model: r.model,
                prediction: r.prediction,
                confidence: r.confidence,
                probability_malignant: r.probability_malignant,
                iris_method: r.iris_method,
            },
            original_filename: entry.file.name,
            annotation_lines: entry.annotator.roundedLines(),
        };

        const form = new FormData();
        form.append("record", JSON.stringify(record));
        form.append("original", entry.file, entry.file.name);
        form.append("gmar", base64ToBlob(r.heatmap, "image/jpeg"), "gmar.jpg");
        const annotated = await entry.annotator.toPNG();
        if (annotated) form.append("annotated", annotated, "annotated.png");

        let res;
        try {
            res = await fetch("/api/evaluations", { method: "POST", body: form });
        } catch {
            throw new Error("cannot reach the server");
        }
        const body = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(body.detail || `server error ${res.status}`);
        return body;
    }

    function setLocked(entry, locked) {
        entry.card.querySelectorAll("input, select, textarea").forEach((c) => { c.disabled = locked; });
        entry.clearAnnotationsBtn.disabled = locked || entry.annotator.lines.length === 0;
        entry.annotator.setEnabled(!locked);
        entry.card.classList.toggle("saved", locked);
    }

    function setSaveMessage(text, kind) {
        saveMessage.textContent = text;
        saveMessage.className = "save-message" + (kind ? " " + kind : "");
    }

    // ── Small DOM / data helpers ──────────────────────────────────────────────
    function field(label, control, { required = false, hint = "", counter = 0 } = {}) {
        const wrap = el("label", "field");
        const title = el("span", "field-label", label);
        if (required) title.append(el("span", "required", " *"));
        wrap.append(title, control);
        if (hint) wrap.append(el("span", "field-hint", hint));
        if (counter) {
            const count = el("span", "field-hint", `0 / ${counter}`);
            control.addEventListener("input", () => { count.textContent = `${control.value.length} / ${counter}`; });
            wrap.append(count);
        }
        control.addEventListener("input", () => control.classList.remove("invalid"));
        control.addEventListener("change", () => control.classList.remove("invalid"));
        return wrap;
    }

    function input(type, props = {}) {
        const node = document.createElement("input");
        node.type = type;
        Object.assign(node, props);
        return node;
    }

    function select(options) {
        const node = document.createElement("select");
        node.append(new Option("Select…", ""), ...options.map((o) => new Option(o, o)));
        return node;
    }

    function textarea(maxLength) {
        const node = document.createElement("textarea");
        node.rows = 3;
        node.maxLength = maxLength;
        return node;
    }

    function figure(src, caption) {
        const fig = el("figure");
        const img = document.createElement("img");
        img.src = src;
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

    const dataUrl = (b64) => `data:image/jpeg;base64,${b64}`;

    function base64ToBlob(b64, type) {
        const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
        return new Blob([bytes], { type });
    }

    const pad = (n) => String(n).padStart(2, "0");

    function localDateTimeValue(d) {
        return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
    }

    function isoWithOffset(d) {
        const off = -d.getTimezoneOffset();
        const sign = off >= 0 ? "+" : "-";
        return `${localDateTimeValue(d)}:${pad(d.getSeconds())}${sign}${pad(Math.floor(Math.abs(off) / 60))}:${pad(Math.abs(off) % 60)}`;
    }
})();
