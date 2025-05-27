"""
Object Defect Detection GUI
────────────────────────────────────────────────────────────────────────────
• Ultralytics YOLO v5 / v8 / v11 (.pt)  – FP16 on CUDA
• TorchVision Faster-R-CNN / SSD (.pth) – auto-detect class count
• Image formats  JPG • PNG • BMP • WEBP • AVIF
• Thread-safe camera capture  |  greedy IoU duplicate-box filter
• Live record (r), screenshot (s), reset zoom (i/I),
  save original video file (a)
• “Open Image” panel accepts drag-and-drop *and* pasted HTTP/HTTPS URLs
• Main menu shows model filename + backend and an **Unload Weights** button
(May 2025)
"""
from __future__ import annotations
import os, pathlib, threading, queue, shutil, tempfile, urllib.request, mimetypes
if os.name == "nt": pathlib.PosixPath = pathlib.WindowsPath          # pickle fix

# ── std / deps ──────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
try: from tkinterdnd2 import TkinterDnD, DND_FILES; HAVE_DND = True
except ImportError: HAVE_DND = False

import cv2, torch, numpy as np
from PIL import Image, ImageTk
import pillow_avif                                            # AVIF plugin

try: from ultralytics import YOLO                              # v8 / v11
except ImportError: YOLO = None

from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    ssd300_vgg16,
)

# ── CONFIG ──────────────────────────────────────────────────────────
BG, DEVICE = "#2b2b2b", ("cuda" if torch.cuda.is_available() else "cpu")
CAP_FLAGS  = cv2.CAP_DSHOW if os.name == "nt" else 0
MIN_Z, MAX_Z = 1.0, 4.0
IOU_TH, SCORE_TH = 0.40, 0.50  #0.50, 0.35
QUEUE_SIZE, CAMERA_RANGE = 1, range(0, 11)
# ── YOLO-specific real-time hygiene ────────────────────────────────
YOLO_CONF   = 0.50      # drop detections whose confidence < 0.50
YOLO_BIG_TH = 0.65      # if a box covers >65 % of the frame but conf < 0.80 → drop

# ── IoU + annotation ───────────────────────────────────────────────
def _iou(a,b):
    xA,yA = max(a[0],b[0]), max(a[1],b[1])
    xB,yB = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,xB-xA)*max(0,yB-yA)
    return 0.0 if not inter else inter/((a[2]-a[0])*(a[3]-a[1]) +
                                        (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)

def annotate(img, boxes, lbls, scores, names):
    """
    Draw boxes & labels.
      • label is kept inside its own box *or* nudged until it no longer
        overlaps any previously-drawn label.
    """
    h, w = img.shape[:2]
    occupied = []                                       # list of drawn label-rects
    for (x1, y1, x2, y2), c, s in zip(boxes, lbls, scores):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        t = max(1, int(min(h, w) * .002)); pad = t
        clr = (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), clr, t)

        # ­— prepare text —
        txt = f"{names[int(c)] if int(c) < len(names) else int(c)} {s:.2f}"
        fs  = min(h, w) / 500
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, t)

        # first try: inside-top-left
        xt, yt = x1 + pad, y1 + th + pad
        top, bot = y1 + pad, yt + bl

        # flip to inside-bottom if needed
        if yt + bl + pad > y2:
            yt = y2 - pad - bl
            top, bot = yt - th - bl, y2 - pad

        # fallback above box
        if top < 0:
            yt = y1 - pad
            top, bot = yt - th - bl - pad, yt + bl + pad

        # horizontal clamp
        if xt + tw + pad > w:
            xt = w - tw - pad

        # ­— LAST RESORT: nudge downward until no overlap with earlier labels —
        step = th + bl + 2 * pad
        while any(not (xt + tw + pad < ox or xt - pad > oxw or
                       top > oby or bot < oy)               # intersects ?
                   for ox, oy, oxw, oby in occupied):
            yt += step
            top, bot = yt - th - bl - pad, yt + bl + pad
            if bot > h:                # out of image – give up
                break

        # draw
        cv2.rectangle(img, (xt - pad, top), (xt + tw + pad, bot), clr, cv2.FILLED)
        cv2.putText(img, txt, (xt, yt), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), t, cv2.LINE_AA)
        occupied.append((xt - pad, top, xt + tw + pad, bot))
    return img

# ── Detector ───────────────────────────────────────────────────────
class Detector:
    def __init__(self, weight: str | pathlib.Path):
        w = str(weight)
        self.tf = transforms.Compose([transforms.ToTensor()])
        self.fp16 = False

        # ------------------------------------------------------------
        # 1)  *.pt  →  try legacy-YOLOv5 first, then Ultralytics v8/v11
        # ------------------------------------------------------------
        if w.endswith(".pt"):
            if self._load_legacy(w):             # <── your v5 checkpoints land here
                ...
            elif self._load_yolo(w):             # v8 / v11
                ...
            else:
                raise RuntimeError("Unsupported YOLO .pt checkpoint")

        elif "ssd" in w.lower():
            self._load_ssd(w)
        elif w.endswith(".pth") or "faster" in w.lower():
            self._load_frcnn(w)
        else:
            raise RuntimeError("Unsupported weight")

        if DEVICE == "cuda" and self.model_type.startswith("yolo"):
            try:
                self.model.half(); self.fp16 = True
            except AttributeError:
                ...
        self.model.to(DEVICE).eval()
        self.filename = pathlib.Path(w).name

    # ---- loaders --------------------------------------------------
    def _load_yolo(self,p):
        if YOLO is None: return False
        try:self.model_type="yolo"; self.model=YOLO(p); self.names=self.model.model.names; return True
        except Exception: return False
    def _load_legacy(self,p):
        try:self.model_type="yolov5-hub"; self.model=torch.hub.load("ultralytics/yolov5","custom",path=p)
        except Exception:
            try:import yolov5; self.model_type="yolov5-pkg"; self.model=yolov5.load(p)
            except Exception: return False
        self.names=self.model.names; return True
    def _load_ssd(self,p):
        self.model_type="ssd"; self.model=ssd300_vgg16(weights=None)
        ck=torch.load(p,map_location=DEVICE); st=ck.get("model_state_dict") or ck.get("state_dict") or ck
        self.model.load_state_dict(st,strict=False)
        self.names=ck.get("class_names") or [f"cls_{i}" for i in range(self.model.head.classification_head.num_classes)]
    def _load_frcnn(self,p):
        self.model_type="fasterrcnn"
        ck=torch.load(p,map_location=DEVICE)
        st=ck.get("model_state_dict") or ck.get("state_dict") or ck
        # --- auto-detect number of classes from checkpoint ---
        nc = 91
        for k in ("roi_heads.box_predictor.cls_score.weight",
                  "box_predictor.cls_score.weight"):
            if k in st:
                nc = st[k].shape[0]; break
        self.model = fasterrcnn_resnet50_fpn(weights=None, num_classes=nc)
        miss = self.model.load_state_dict(st, strict=False)
        if miss.missing_keys or miss.unexpected_keys:
            print("FasterRCNN – ignored keys:", miss)
        self.names = ck.get("class_names") or [f"cls_{i}" for i in range(nc)]

    # ---- inference ------------------------------------------------
    @torch.inference_mode()
    def infer(self,bgr):
        if self.model_type.startswith("yolo"): b,l,s=self._pyolo(bgr)
        elif self.model_type.startswith("yolov5"): b,l,s=self._pv5(bgr)
        else: b,l,s=self._ptv(bgr)
        b,l,s=self._nms(b,l,s)
        return annotate(bgr,b,l,s,self.names)
    @torch.inference_mode()
    def _pyolo(self, bgr: np.ndarray):
        """
        Works for all Ultralytics flavours:

        • v8 / v11  →  .predict()
        • legacy YOLOv5 AutoShape  →  call model directly
        """

        try:                                # modern Ultralytics
            res = self.model.predict(
                bgr, conf=0.7, device=DEVICE, verbose=False
            )[0] # conf=YOLO_CONF 
            return (res.boxes.xyxy.cpu().numpy(),
                    res.boxes.cls.cpu().numpy(),
                    res.boxes.conf.cpu().numpy())

        except (AttributeError, TypeError):  # ← now also catches the “embed” TypeError
            # AutoShape or mismatched forward() – fall back to __call__
            res  = self.model(bgr)                       # list-like Results
            xyxy = res.xyxy[0].cpu().numpy()
            return xyxy[:, :4], xyxy[:, 5].astype(int), xyxy[:, 4]
    def _pv5(self,bgr):
        a=self.model(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).xyxy[0].cpu().numpy()
        return a[:,:4],a[:,5].astype(int),a[:,4]
    def _ptv(self,bgr):
        t=self.tf(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).to(DEVICE); t=t.half() if self.fp16 else t
        o=self.model([t])[0]; m=o["scores"]>SCORE_TH
        return o["boxes"][m].cpu().numpy(),o["labels"][m].cpu().numpy(),o["scores"][m].cpu().numpy()
    def _nms(self, boxes, labels, scores):
        """
        Remove duplicates irrespective of class:
        keep the highest-confidence box if IoU > IOU_TH (0.50 by default).

        This runs on the CPU-side NumPy arrays returned by *_pyolo(), *_pv5()
        or *_ptv() and therefore works for EVERY backend (YOLO v5 / v8 / v11,
        Faster-RCNN, SSD).
        """
        if len(boxes) == 0:
            return boxes, labels, scores

        order = scores.argsort()[::-1]          # high-conf → low-conf
        keep  = []

        for i in order:
            if scores[i] < SCORE_TH:            # drop very low scores
                continue
            # ---- compare against boxes we’ve already kept -------------
            if any(_iou(boxes[i], boxes[j]) > IOU_TH for j in keep):
                continue                        # overlaps → skip
            keep.append(i)

        keep = np.asarray(keep, dtype=int)
        return boxes[keep], labels[keep], scores[keep]




# ── threaded worker (drops stale frames) ───────────────────────────
class VideoWorker(threading.Thread):
    def __init__(self,cap,det,q):
        super().__init__(daemon=True); self.cap,self.det,self.q=cap,det,q; self.stop_evt=threading.Event()
    def run(self):
        while not self.stop_evt.is_set():
            ok,frm=self.cap.read()
            if not ok: self.stop_evt.set(); break
            rgb=cv2.cvtColor(self.det.infer(frm),cv2.COLOR_BGR2RGB)
            while not self.q.empty():
                try:self.q.get_nowait()
                except queue.Empty: break
            self.q.put(rgb)
        self.cap.release()
    def stop(self): self.stop_evt.set()

BaseTk = TkinterDnD.Tk if HAVE_DND else tk.Tk

# ── GUI ─────────────────────────────────────────────────────────────
BaseTk = TkinterDnD.Tk if HAVE_DND else tk.Tk

class App:
    def __init__(self, root: BaseTk):
        self.root = root
        root.title("Object Defect Detection")
        root.configure(bg=BG); root.geometry("960x640")

        self.detector: Detector | None = None
        self.zoom, self.ox, self.oy = MIN_Z, 0, 0
        self.current: np.ndarray | None = None
        self.canvas = None

        self.worker: VideoWorker | None = None
        self.frame_q: queue.Queue | None = None
        self.record, self.writer = False, None
        self.src_path, self.fps = None, 30

        self._style(); self._home()

    # -------- style / home menu ------------------------------------
    def _style(self):
        s = ttk.Style(self.root); s.theme_use("clam")
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground="white")
        s.configure("Header.TLabel",
                    font=("Helvetica", 20, "bold"),
                    background=BG, foreground="white")
        s.configure("TButton", font=("Helvetica", 12), padding=8)

    def _home(self):
        self._clear(); self._unbind_keys()
        f = ttk.Frame(self.root, padding=40); f.pack(expand=True, fill="both")
        ttk.Label(f, text="Object Defect Detection",
                  style="Header.TLabel").pack()
        ttk.Label(f, text=self._stxt(),
                  foreground=self._sclr()).pack()

        if self.detector:
            ttk.Label(f,
                      text=f"{self.detector.filename}  "
                           f"({self.detector.model_type})",
                      foreground="lightgray").pack(pady=(0, 20))
        else:
            ttk.Frame(f).pack(pady=(0, 20))        # spacer

        buttons = [("Find Weights", self._pick)]
        if self.detector:
            buttons.append(("Unload Weights", self._unload))
        buttons += [("Open Camera", self._cam),
                    ("Open Image", self._open_img_panel),
                    ("Open Video", self._vid),
                    ("Exit", self.root.destroy)]
        for t, fn in buttons:
            ttk.Button(f, text=t, command=fn,
                       width=25).pack(pady=6)

    def _clear(self): [w.destroy() for w in self.root.winfo_children()]
    def _stxt(self):  return "Weights loaded ✅" if self.detector else "No weights ❌"
    def _sclr(self):  return "green" if self.detector else "red"

    # -------- weight handling --------------------------------------
    def _pick(self):
        p = filedialog.askopenfilename(
            filetypes=[("PyTorch", "*.pt *.pth")])
        if not p:
            return
        try:
            self.detector = Detector(p)
            messagebox.showinfo("Model",
                                f"Loaded {self.detector.model_type}")
        except Exception as e:
            self.detector = None
            messagebox.showerror("Load error", str(e))
        self._home()

    def _unload(self):
        self.detector = None
        messagebox.showinfo("Weights", "Weights cleared from memory.")
        self._home()

    def _need_weights(self):
        if self.detector:
            return True
        messagebox.showwarning("No weights", "Please load weights first.")
        return False

    # =================================================================
    # CAMERA  (unchanged from previous build)
    # =================================================================
    def _cam(self):
        if not self._need_weights():
            return
        cams = [i for i in CAMERA_RANGE if
                cv2.VideoCapture(i, CAP_FLAGS).isOpened()]
        if not cams:
            messagebox.showerror("Error", "No camera found"); return
        cam = (cams[0] if len(cams) == 1 else
               simpledialog.askinteger(
                   "Camera", f"Devices: {cams}\nSelect index:",
                   initialvalue=cams[0]))
        if cam is None:
            return
        cap = cv2.VideoCapture(cam, CAP_FLAGS)
        if not cap.isOpened():
            messagebox.showerror("Error",
                                 f"Cannot open camera {cam}"); return
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.src_path = None
        self._start_stream(cap, show_rec=True, show_archive=False,
                           allow_switch=len(cams) > 1, cam_list=cams)

    # =================================================================
    # OPEN IMAGE PANEL  (drag-drop / URL-paste / browse)
    # =================================================================
    def _open_img_panel(self):
        if not self._need_weights():
            return
        self._clear(); self._unbind_keys()
        self.zoom, self.ox, self.oy = MIN_Z, 0, 0

        bar = ttk.Frame(self.root); bar.pack(fill="x")
        ttk.Button(bar, text="← Back",
                   command=self._home).pack(side="left")

        panel = ttk.Frame(self.root, padding=40)
        panel.pack(expand=True, fill="both")
        ttk.Label(panel,
                  text=("Drop image file here,\n"
                        "press Ctrl+V to paste URL,\n"
                        "or click “Browse”"),
                  style="Header.TLabel", justify="center").pack(expand=True)
        ttk.Button(panel, text="Browse…",
                   command=self._browse_image).pack()

        # clipboard paste
        self.root.bind_all("<Control-v>", self._paste_url)
        self.root.bind_all("<Command-v>", self._paste_url)     # mac

        # drag-and-drop
        if HAVE_DND:
            panel.drop_target_register(DND_FILES)
            panel.dnd_bind("<<Drop>>", self._drop_file)
        else:
            ttk.Label(panel,
                      text="(drag-and-drop unavailable – "
                           "install tkinterdnd2)",
                      background=BG, foreground="yellow").pack()

    def _browse_image(self):
        p = filedialog.askopenfilename(
            filetypes=[("Images",
                        "*.jpg *.jpeg *.png *.bmp *.webp *.avif")])
        if p:
            self._load_image_from_path(p)

    def _drop_file(self, e):
        path = e.data.strip().strip("{}").split()[0]
        self._load_image_from_path(path)

    def _paste_url(self, _e):
        url = self.root.clipboard_get().strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            return
        try:
            with urllib.request.urlopen(url) as resp:
                data = resp.read(); mime = resp.info().get_content_type()
        except Exception as ex:
            messagebox.showerror("Download error", str(ex)); return
        ext = mimetypes.guess_extension(mime) or ".jpg"
        tmp = pathlib.Path(tempfile.gettempdir()) / f"dl{os.getpid()}{ext}"
        tmp.write_bytes(data)
        self._load_image_from_path(str(tmp))

    def _load_image_from_path(self, path: str):
        bgr = cv2.imread(path) if cv2.haveImageReader(path) else None
        if bgr is None and path.lower().endswith(".avif"):
            bgr = cv2.cvtColor(
                np.array(Image.open(path).convert("RGB")),
                cv2.COLOR_RGB2BGR)
        if bgr is None:
            messagebox.showerror("Error", "Unsupported image."); return
        self.current = cv2.cvtColor(
            self.detector.infer(bgr), cv2.COLOR_BGR2RGB)

        # remove clipboard binds
        self.root.unbind_all("<Control-v>")
        self.root.unbind_all("<Command-v>")
        self._show_still()

    # =================================================================
    # OPEN VIDEO FILE
    # =================================================================
    def _vid(self):
        if not self._need_weights():
            return
        p = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov")])
        if not p:
            return
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open video."); return
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.src_path = p
        self._start_stream(cap, show_rec=True, show_archive=True)

    # =================================================================
    # STREAM / WORKER / GUI TOOLBAR
    # =================================================================
    def _start_stream(self, cap, *, show_rec, show_archive,
                      allow_switch=False, cam_list=None):
        self._clear(); self.zoom, self.ox, self.oy = MIN_Z, 0, 0
        bar = ttk.Frame(self.root); bar.pack(fill="x")
        ttk.Button(bar, text="← Back",
                   command=self._stop_stream).pack(side="left")
        for t, f in (("Zoom In", lambda: self._zoom(self.zoom*1.2)),
                     ("Zoom Out", lambda: self._zoom(self.zoom/1.2)),
                     ("Reset (I)", lambda: self._zoom(MIN_Z))):
            ttk.Button(bar, text=t, command=f).pack(side="left")
        if show_rec:
            self.rec_btn = ttk.Button(bar, text="● Rec",
                                      command=self._toggle_rec)
            self.rec_btn.pack(side="left")
        if show_archive:
            ttk.Button(bar, text="Save original (a)",
                       command=self._save_original).pack(side="left")
        ttk.Button(bar, text="Screenshot",
                   command=self._screenshot).pack(side="left")
        if allow_switch:
            ttk.Button(bar, text="Switch Cam",
                       command=lambda: self._switch_cam(cam_list)).pack(side="left")

        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._bind_nav()
        self._bind_keys(stream=True, archive=show_archive)

        self.frame_q = queue.Queue(QUEUE_SIZE)
        self.worker = VideoWorker(cap, self.detector, self.frame_q)
        self.worker.start(); self._poll_frames()

    def _poll_frames(self):
        if self.worker and not self.worker.stop_evt.is_set():
            try:
                self.current = self.frame_q.get_nowait()
                self._draw()
            except queue.Empty:
                ...
            if self.record and self.writer and self.current is not None:
                self.writer.write(cv2.cvtColor(
                    self.current, cv2.COLOR_RGB2BGR))
            self.root.after(1, self._poll_frames)
        else:
            self._stop_stream()

    def _stop_stream(self):
        if self.record:
            self._toggle_rec()
        if self.worker:
            self.worker.stop(); self.worker.join(); self.worker = None
        self._unbind_keys(); self._home()

    def _switch_cam(self, cams):
        if not self.worker:
            return
        idx = cams.index(int(self.worker.cap.get(cv2.CAP_PROP_DEVICE)))
        new = cams[(idx+1) % len(cams)]
        self.worker.stop(); self.worker.join()
        cap = cv2.VideoCapture(new, CAP_FLAGS)
        if not cap.isOpened():
            messagebox.showerror("Error",
                                 f"Cannot open camera {new}"); return
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self._start_stream(cap, show_rec=True,
                           show_archive=False,
                           allow_switch=True, cam_list=cams)

    # === record / screenshot / save-original ========================
    def _toggle_rec(self, *_):
        if not self.record:
            if self.current is None:
                return
            p = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")])
            if not p:
                return
            fourcc = cv2.VideoWriter_fourcc(
                *('mp4v' if p.lower().endswith(".mp4") else 'XVID'))
            h, w = self.current.shape[:2]
            self.writer = cv2.VideoWriter(p, fourcc, self.fps, (w, h))
            if not self.writer.isOpened():
                messagebox.showerror("Error", "Writer open failed."); return
            self.record = True; self.rec_btn.config(text="■ Stop")
        else:
            self.record = False; self.rec_btn.config(text="● Rec")
            if self.writer:
                self.writer.release(); self.writer = None

    def _save_original(self, *_):
        if not self.src_path:
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=pathlib.Path(self.src_path).suffix,
            initialfile=pathlib.Path(self.src_path).name)
        if not dest:
            return
        try:
            shutil.copy2(self.src_path, dest)
            messagebox.showinfo("Saved", f"Copied to\n{dest}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _screenshot(self, *_):
        if self.current is None:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"),
                       ("JPEG", "*.jpg *.jpeg")])
        if p:
            cv2.imwrite(p, cv2.cvtColor(
                self.current, cv2.COLOR_RGB2BGR))

    # =================================================================
    # STILL IMAGE VIEWER
    # =================================================================
    def _show_still(self):
        self._clear()
        bar = ttk.Frame(self.root); bar.pack(fill="x")
        for t, f in (("← Back", self._home),
                     ("Zoom In", lambda: self._zoom(self.zoom*1.2)),
                     ("Zoom Out", lambda: self._zoom(self.zoom/1.2)),
                     ("Reset (I)", lambda: self._zoom(MIN_Z)),
                     ("Save (s)", self._screenshot)):
            ttk.Button(bar, text=t, command=f).pack(side="left")
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.zoom, self.ox, self.oy = MIN_Z, 0, 0
        self._bind_nav(); self._bind_keys(stream=False, archive=False); self._draw()

    # =================================================================
    # Navigation / drawing
    # =================================================================
    def _bind_nav(self):
        self.canvas.bind("<MouseWheel>",
                         lambda e: self._zoom(self.zoom*(1.2 if e.delta>0 else 1/1.2)))
        self.canvas.bind("<ButtonPress-1>", lambda e: setattr(self, "_p", (e.x, e.y)))
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<Configure>", lambda _e: self._draw())

    def _drag(self, e):
        if self.zoom == MIN_Z:
            return
        dx, dy = e.x - self._p[0], e.y - self._p[1]; self._p = (e.x, e.y)
        self.ox += dx; self.oy += dy; self._draw()

    def _zoom(self, z):
        if not MIN_Z <= z <= MAX_Z:
            return
        self.zoom = z
        if z == MIN_Z:
            self.ox = self.oy = 0
        self._draw()

    def _draw(self):
        if self.current is None or not self.canvas:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        ih, iw = self.current.shape[:2]
        scale = min(cw / iw, ch / ih) * self.zoom
        disp = cv2.resize(self.current,
                          (int(iw * scale), int(ih * scale)))
        img = ImageTk.PhotoImage(Image.fromarray(disp))
        self.canvas.delete("all")
        self.canvas.create_image(
            cw // 2 + self.ox, ch // 2 + self.oy, image=img)
        self.canvas.image = img

    # =================================================================
    # Global key bindings
    # =================================================================
    def _bind_keys(self, *, stream, archive):
        self._unbind_keys()
        self.root.bind("s", self._screenshot)
        self.root.bind("i", lambda *_: self._zoom(MIN_Z))
        self.root.bind("I", lambda *_: self._zoom(MIN_Z))
        if stream:
            self.root.bind("r", self._toggle_rec)
        if archive:
            self.root.bind("a", self._save_original)

    def _unbind_keys(self):
        for k in ("s", "r", "a", "i", "I"):
            self.root.unbind(k)

# ── main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = BaseTk()
    App(root)
    root.mainloop()
