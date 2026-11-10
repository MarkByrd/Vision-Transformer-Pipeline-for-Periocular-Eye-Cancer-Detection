/*
 * Annotator: draw freehand lines over a copy of an image (spec FR-06 / FR-07).
 *
 *   const a = new Annotator(container, imageUrl, fallbackUrl);
 *   a.lines        -> [[ [x, y], ... ], ...]  points normalised to 0-1
 *   await a.toPNG()  -> Blob of the image with the lines burned in (null if no lines)
 *   a.clear(), a.setEnabled(false)
 *
 * The source image is only drawn onto a canvas; the uploaded file is never changed.
 */
(function () {
    "use strict";

    const MAX_SIDE = 2048;          // cap canvas size for very large photos
    const LINE_COLOUR = "#facc15";  // yellow: visible on skin, iris and sclera

    class Annotator {
        constructor(container, src, fallbackSrc) {
            this.lines = [];
            this.enabled = true;
            this.drawing = null;
            this.canvas = document.createElement("canvas");
            this.canvas.className = "annotation-canvas";
            this.ctx = this.canvas.getContext("2d");
            container.append(this.canvas);

            this.image = new Image();
            this.image.onload = () => this._init();
            // Browsers can't decode some formats (e.g. TIFF); fall back to the
            // server's cropped preview so annotation is still possible.
            this.image.onerror = () => {
                if (fallbackSrc && this.image.src !== fallbackSrc) this.image.src = fallbackSrc;
            };
            this.image.src = src;

            this.canvas.addEventListener("pointerdown", (e) => this._start(e));
            this.canvas.addEventListener("pointermove", (e) => this._move(e));
            this.canvas.addEventListener("pointerup", () => this._end());
            this.canvas.addEventListener("pointercancel", () => this._end());
        }

        _init() {
            const { naturalWidth: w, naturalHeight: h } = this.image;
            const scale = Math.min(1, MAX_SIDE / Math.max(w, h));
            this.canvas.width = Math.round(w * scale);
            this.canvas.height = Math.round(h * scale);
            this.lineWidth = Math.max(3, Math.round(Math.max(this.canvas.width, this.canvas.height) / 150));
            this._redraw();
        }

        _point(e) {
            const r = this.canvas.getBoundingClientRect();
            const clamp = (v) => Math.min(1, Math.max(0, v));
            return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)];
        }

        _start(e) {
            if (!this.enabled || !this.canvas.width) return;
            e.preventDefault();
            this.canvas.setPointerCapture(e.pointerId);
            this.drawing = [this._point(e)];
            this.lines.push(this.drawing);
            this._redraw();
        }

        _move(e) {
            if (!this.drawing) return;
            this.drawing.push(this._point(e));
            this._redraw();
        }

        _end() {
            if (!this.drawing) return;
            this.drawing = null;
            this.onchange && this.onchange();
        }

        _redraw() {
            const { ctx, canvas } = this;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(this.image, 0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = ctx.fillStyle = LINE_COLOUR;
            ctx.lineWidth = this.lineWidth;
            ctx.lineCap = ctx.lineJoin = "round";
            for (const line of this.lines) {
                const pts = line.map(([x, y]) => [x * canvas.width, y * canvas.height]);
                if (pts.length === 1) {  // a click: draw a dot
                    ctx.beginPath();
                    ctx.arc(pts[0][0], pts[0][1], this.lineWidth, 0, Math.PI * 2);
                    ctx.fill();
                    continue;
                }
                ctx.beginPath();
                ctx.moveTo(...pts[0]);
                for (const p of pts.slice(1)) ctx.lineTo(...p);
                ctx.stroke();
            }
        }

        clear() {
            this.lines = [];
            this._redraw();
            this.onchange && this.onchange();
        }

        setEnabled(enabled) {
            this.enabled = enabled;
            this.canvas.classList.toggle("locked", !enabled);
        }

        roundedLines() {
            return this.lines.map((l) => l.map(([x, y]) => [+x.toFixed(4), +y.toFixed(4)]));
        }

        toPNG() {
            if (!this.lines.length) return Promise.resolve(null);
            return new Promise((resolve) => this.canvas.toBlob(resolve, "image/png"));
        }
    }

    window.Annotator = Annotator;
})();
