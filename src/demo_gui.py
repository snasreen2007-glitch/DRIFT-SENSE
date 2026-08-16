"""
demo_gui.py
=============
Live hackathon demo GUI for DRIFT-SENSE.

Browse to a reference patch + search image, click "Run Detection", and
see the result (annotated + zoomed) directly in the window, along with
confidence, timing, and debug numbers -- no terminal reading required
mid-demo.

Usage:
    python src/demo_gui.py
    python src/demo_gui.py --weights weights/driftsense_cnn.pt
"""

import argparse
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from drift_sense_pipeline import DriftSenseDetector


class DriftSenseDemoApp:
    def __init__(self, root, weights_path=None):
        self.root = root
        self.root.title("DRIFT-SENSE -- Live Demo")
        self.root.geometry("1050x680")
        self.root.configure(bg="#1e1e1e")

        self.reference_path = None
        self.search_path = None
        self.detector = None
        self.weights_path = weights_path

        self._build_ui()
        self._load_detector_async()

    # ---------------- UI ----------------
    def _build_ui(self):
        FG = "#e8e8e8"
        BG = "#1e1e1e"
        ACCENT = "#2ecc71"

        header = tk.Label(self.root, text="DRIFT-SENSE",
                           font=("Helvetica", 22, "bold"), fg=ACCENT, bg=BG)
        header.pack(pady=(12, 0))
        subheader = tk.Label(self.root, text="Hybrid AI Navigation-Error Recovery -- Live Demo",
                              font=("Helvetica", 11), fg="#999999", bg=BG)
        subheader.pack(pady=(0, 10))

        # --- top control bar ---
        controls = tk.Frame(self.root, bg=BG)
        controls.pack(fill="x", padx=20)

        self.ref_btn = tk.Button(controls, text="1. Choose Reference Patch",
                                  command=self.choose_reference, bg="#333333", fg=FG,
                                  activebackground="#444444", relief="flat", padx=12, pady=6)
        self.ref_btn.grid(row=0, column=0, padx=(0, 10), pady=4, sticky="w")

        self.search_btn = tk.Button(controls, text="2. Choose Search Image",
                                     command=self.choose_search, bg="#333333", fg=FG,
                                     activebackground="#444444", relief="flat", padx=12, pady=6)
        self.search_btn.grid(row=0, column=1, padx=(0, 10), pady=4, sticky="w")

        self.run_btn = tk.Button(controls, text="Run Detection", command=self.run_detection,
                                  bg=ACCENT, fg="#0b0b0b", activebackground="#27ae60",
                                  relief="flat", padx=18, pady=6, font=("Helvetica", 10, "bold"),
                                  state="disabled")
        self.run_btn.grid(row=0, column=2, padx=(20, 0), pady=4, sticky="w")

        self.status_label = tk.Label(self.root, text="Loading detector...", fg="#ffaa00", bg=BG,
                                      font=("Helvetica", 10))
        self.status_label.pack(pady=(6, 6))

        # --- image display area ---
        images_frame = tk.Frame(self.root, bg=BG)
        images_frame.pack(fill="both", expand=True, padx=20, pady=6)

        ref_col = tk.Frame(images_frame, bg=BG)
        ref_col.pack(side="left", padx=10)
        tk.Label(ref_col, text="Reference", fg=FG, bg=BG, font=("Helvetica", 10, "bold")).pack()
        self.ref_canvas = tk.Label(ref_col, bg="#0b0b0b", width=180, height=180)
        self.ref_canvas.pack(pady=4)

        result_col = tk.Frame(images_frame, bg=BG)
        result_col.pack(side="left", padx=10, fill="both", expand=True)
        tk.Label(result_col, text="Detection Result (zoomed)", fg=FG, bg=BG,
                 font=("Helvetica", 10, "bold")).pack()
        self.result_canvas = tk.Label(result_col, bg="#0b0b0b", width=420, height=420)
        self.result_canvas.pack(pady=4)

        # --- results text panel ---
        self.results_text = tk.Label(self.root, text="", fg=FG, bg=BG, justify="left",
                                      font=("Consolas", 11), anchor="w")
        self.results_text.pack(fill="x", padx=24, pady=(6, 14))

    def _load_detector_async(self):
        self.root.after(100, self._load_detector)

    def _load_detector(self):
        try:
            self.detector = DriftSenseDetector(cnn_weights_path=self.weights_path)
            weight_note = f" (trained weights: {Path(self.weights_path).name})" if self.weights_path else " (untrained CNN)"
            self.status_label.config(text=f"Detector ready{weight_note}", fg="#2ecc71")
        except Exception as e:
            self.status_label.config(text=f"Failed to load detector: {e}", fg="#e74c3c")
            return
        self._update_run_button()

    # ---------------- file pickers ----------------
    def choose_reference(self):
        path = filedialog.askopenfilename(title="Choose reference patch image",
                                           filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.reference_path = path
            self._show_thumbnail(path, self.ref_canvas, size=(170, 170))
            self._update_run_button()

    def choose_search(self):
        path = filedialog.askopenfilename(title="Choose search image",
                                           filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.search_path = path
            self._update_run_button()

    def _update_run_button(self):
        ready = self.reference_path and self.search_path and self.detector is not None
        self.run_btn.config(state="normal" if ready else "disabled")

    def _show_thumbnail(self, path, widget, size):
        img = Image.open(path).convert("L").resize(size, Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        widget.configure(image=photo)
        widget.image = photo  # keep reference

    # ---------------- detection ----------------
    def run_detection(self):
        self.status_label.config(text="Running detection...", fg="#ffaa00")
        self.root.update_idletasks()

        reference = cv2.imread(self.reference_path, cv2.IMREAD_GRAYSCALE)
        search_gray = cv2.imread(self.search_path, cv2.IMREAD_GRAYSCALE)
        search_bgr = cv2.imread(self.search_path, cv2.IMREAD_COLOR)

        if reference is None or search_gray is None:
            messagebox.showerror("Error", "Could not read one of the selected images.")
            return

        t0 = time.time()
        result = self.detector.locate(reference, search_gray)
        elapsed = time.time() - t0

        # draw annotated + zoomed result
        color = (0, 200, 0) if result.reliable else (0, 0, 255)
        x, y = int(round(result.x)), int(round(result.y))
        annotated = search_bgr.copy()
        half = reference.shape[0] // 2
        cv2.rectangle(annotated, (x - half, y - half), (x + half, y + half), color, 2)
        cv2.drawMarker(annotated, (x, y), color, markerType=cv2.MARKER_CROSS,
                        markerSize=24, thickness=2)

        zoom_half = max(60, reference.shape[0])
        h, w = annotated.shape[:2]
        y0, y1 = max(0, y - zoom_half), min(h, y + zoom_half)
        x0, x1 = max(0, x - zoom_half), min(w, x + zoom_half)
        zoom_crop = annotated[y0:y1, x0:x1]
        zoom_crop_rgb = cv2.cvtColor(zoom_crop, cv2.COLOR_BGR2RGB)
        zoom_img = Image.fromarray(zoom_crop_rgb).resize(
            (zoom_crop.shape[1] * 3, zoom_crop.shape[0] * 3), Image.NEAREST)
        photo = ImageTk.PhotoImage(zoom_img)
        self.result_canvas.configure(image=photo)
        self.result_canvas.image = photo

        status_word = "RELIABLE" if result.reliable else "LOW CONFIDENCE"
        status_color = "#2ecc71" if result.reliable else "#e74c3c"
        self.status_label.config(text=f"Detection complete -- {status_word}", fg=status_color)

        info = (
            f"Predicted location : ({result.x:.2f}, {result.y:.2f})\n"
            f"Confidence          : {result.confidence:.3f}   [{result.reason}]\n"
            f"Reliable             : {result.reliable}\n"
            f"Inference time       : {elapsed * 1000:.1f} ms\n"
            f"Coarse score / Fine NCC : {result.debug.get('best_coarse_score', 0):.3f} / "
            f"{result.debug.get('best_fine_ncc', 0):.3f}   "
            f"(runner-up NCC: {result.debug.get('second_best_fine_ncc', 0):.3f})\n"
            f"Estimated rotation / scale : {result.debug.get('rotation_deg', 0)}deg / "
            f"{result.debug.get('scale', 1):.2f}x"
        )
        self.results_text.config(text=info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    root = tk.Tk()
    app = DriftSenseDemoApp(root, weights_path=args.weights)
    root.mainloop()


if __name__ == "__main__":
    main()
