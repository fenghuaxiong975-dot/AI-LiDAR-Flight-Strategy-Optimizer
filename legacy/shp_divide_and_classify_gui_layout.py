import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import sys
import time
import threading
import laspy
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import pandas as pd
import torch
from pct.model import Pct


class LiDAROptimizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-assisted LiDAR flight strategy optimizer")
        self.root.geometry("1020x650")
        self.root.resizable(False, False)

        # 图片样式配色方案
        self.bg_color = "#d3d3d3"
        self.frame_color = "#ededed"
        self.accent_color = "#737373"
        self.text_color = "#1f1f1f"
        self.border_color = "#5f5f5f"
        self.header_color = "#d3d3d3"
        self.header_text = "#111111"
        self.button_bg = "#f3f3f3"
        self.button_hover = "#dfdfdf"
        self.success_color = "#f3f3f3"
        self.success_hover = "#dfdfdf"
        self.header_bar_color = "#d7d7d7"
        self.value_color = "#c84a4a"

        self.root.configure(bg=self.bg_color)

        # 上部分：生育期检测 & 飞行参数推荐
        self.lidar_file = ""
        self.vector_file = ""
        base_dir = (getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
                    if getattr(sys, 'frozen', False)
                    else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_file = os.path.join(base_dir, 'models', 'latest_model-new.t7')
        self.output_dir = ""
        self.results = []
        self.post_ratio = 0
        self.is_processing_stage = False

        # 下部分：株高计算（独立）
        self.lidar_file_height = ""
        self.vector_file_height = ""
        self.height_results = []
        self.is_processing_height = False
        self.last_stage_result_file = ""
        self.last_height_result_file = ""

        self.create_widgets()

    def create_widgets(self):
        # 设置进度条样式为绿色
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            'green.Horizontal.TProgressbar',
            background='#8fd17f',
            troughcolor='#d8d8d8',
            bordercolor=self.border_color,
            lightcolor='#8fd17f',
            darkcolor='#8fd17f',
            thickness=14,
        )
        style.map('green.Horizontal.TProgressbar',
                  background=[('active', '#8fd17f')])

        # 标题栏
        title_frame = tk.Frame(self.root, bg=self.header_color, height=70)
        title_frame.pack(fill=tk.X, padx=22, pady=(10, 6))
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text='AI-assisted LiDAR flight strategy optimizer',
            font=('Times New Roman', 28, 'bold italic'),
            bg=self.header_color,
            fg=self.header_text,
        )
        title_label.pack(anchor=tk.W, pady=(2, 0))

        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=22, pady=(0, 18))

        # 左侧导航面板
        self.left_frame = tk.Frame(
            main_frame,
            bg=self.frame_color,
            width=150,
            height=430,
            bd=1,
            relief=tk.SOLID,
            highlightbackground=self.border_color,
            highlightthickness=1,
        )
        self.left_frame.pack(side=tk.LEFT, padx=(0, 18), pady=8, anchor='n')
        self.left_frame.pack_propagate(False)
        self.left_bottom_line = tk.Frame(
            self.left_frame,
            bg=self.border_color,
            height=1,
            bd=0,
            highlightthickness=0,
        )

        # 右侧主内容区域
        self.right_frame = tk.Frame(main_frame, bg=self.bg_color)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=8)

        # ===== 上方：生育期检测 =====
        self.top_card = tk.Frame(
            self.right_frame,
            bg=self.frame_color,
            bd=1,
            relief=tk.SOLID,
            highlightbackground=self.border_color,
            highlightthickness=1,
            padx=14,
            pady=12,
        )
        self.top_card.pack(fill=tk.X)
        self.top_card.grid_columnconfigure(2, weight=1)

        tk.Label(
            self.top_card,
            text='Input',
            font=('Times New Roman', 13, 'bold'),
            bg=self.frame_color,
            fg=self.text_color,
        ).grid(row=0, column=0, rowspan=3, sticky='nw', padx=(0, 16), pady=(2, 0))

        lidar_btn = tk.Button(
            self.top_card,
            text='Lidar data',
            width=12,
            font=('Arial', 10, 'bold'),
            bg=self.button_bg,
            fg=self.text_color,
            activebackground=self.button_hover,
            relief=tk.GROOVE,
            bd=1,
            command=self.select_lidar_file,
        )
        lidar_btn.grid(row=0, column=1, sticky='w', padx=(0, 20), pady=(0, 8))

        vector_btn = tk.Button(
            self.top_card,
            text='Vector file',
            width=12,
            font=('Arial', 10, 'bold'),
            bg=self.button_bg,
            fg=self.text_color,
            activebackground=self.button_hover,
            relief=tk.GROOVE,
            bd=1,
            command=self.select_vector_file,
        )
        vector_btn.grid(row=1, column=1, sticky='w', padx=(0, 20), pady=(0, 8))

        start_stage_btn = tk.Button(
            self.top_card,
            text='Start',
            width=12,
            font=('Times New Roman', 11, 'bold'),
            bg=self.button_bg,
            fg=self.value_color,
            activebackground=self.button_hover,
            relief=tk.GROOVE,
            bd=1,
            command=self.run_stage_detection,
        )
        start_stage_btn.grid(row=2, column=1, sticky='w', padx=(0, 20))

        top_info = tk.Frame(self.top_card, bg=self.frame_color)
        top_info.grid(row=0, column=2, rowspan=3, sticky='nsew')
        top_info.grid_columnconfigure(1, weight=1)

        tk.Label(
            top_info,
            text='progress',
            font=('Times New Roman', 13, 'bold'),
            bg=self.frame_color,
            fg=self.text_color,
        ).grid(row=0, column=0, sticky='w', padx=(0, 12), pady=(0, 10))

        self.progress_bar1 = ttk.Progressbar(
            top_info,
            length=230,
            mode='determinate',
            style='green.Horizontal.TProgressbar',
        )
        self.progress_bar1.grid(row=0, column=1, sticky='w', pady=(0, 10))

        self.stage_label = tk.Label(
            top_info,
            text='Growth stage: --',
            bg=self.frame_color,
            fg=self.text_color,
            font=('Times New Roman', 13, 'bold'),
            anchor='w',
        )
        self.stage_label.grid(row=1, column=0, columnspan=2, sticky='w', pady=(0, 8))

        self.stage_file_label = tk.Label(
            top_info,
            text='Result: waiting',
            bg=self.frame_color,
            fg=self.text_color,
            font=('Times New Roman', 13, 'bold'),
            anchor='w',
        )
        self.stage_file_label.grid(row=2, column=0, columnspan=2, sticky='w')

        # ===== 中间：飞行参数推荐 =====
        self.params_frame = tk.Frame(self.right_frame, bg=self.bg_color)
        self.params_frame.pack(fill=tk.X, pady=16)
        self.params_frame.grid_columnconfigure(0, weight=1)
        self.params_frame.grid_columnconfigure(1, weight=1)

        optimal_frame = tk.Frame(
            self.params_frame,
            bg=self.frame_color,
            bd=1,
            relief=tk.SOLID,
            highlightbackground=self.border_color,
            highlightthickness=1,
        )
        optimal_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))

        optimal_header = tk.Label( 
            optimal_frame,
            text='Accuracy-optimal:',
            font=('Times New Roman', 14, 'bold'),
            bg=self.header_bar_color,
            fg=self.text_color,
            anchor='w',
            padx=10,
            pady=8,
        )
        optimal_header.pack(fill=tk.X)

        optimal_body = tk.Frame(optimal_frame, bg=self.frame_color)
        optimal_body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.optimal_labels = {}
        for param in ['Speed', 'Altitude', 'Overlap rate', 'Scanning angle']:
            row = tk.Frame(optimal_body, bg=self.frame_color)
            row.pack(fill=tk.X, pady=3)
            name_label = tk.Label(
                row,
                text=f'{param}: ',
                bg=self.frame_color,
                fg=self.text_color,
                font=('Times New Roman', 13, 'bold'),
                anchor='w',
            )
            name_label.pack(side=tk.LEFT)
            value_label = tk.Label(
                row,
                text='--',
                bg=self.frame_color,
                fg=self.text_color,
                font=('Times New Roman', 13, 'bold'),
                anchor='w',
            )
            value_label.pack(side=tk.LEFT)
            self.optimal_labels[param] = value_label

        balanced_frame = tk.Frame(
            self.params_frame,
            bg=self.frame_color,
            bd=1,
            relief=tk.SOLID,
            highlightbackground=self.border_color,
            highlightthickness=1,
        )
        balanced_frame.grid(row=0, column=1, sticky='nsew', padx=(10, 0))

        balanced_header = tk.Label(
            balanced_frame,
            text='Accuracy efficiency balanced:',
            font=('Times New Roman', 14, 'bold'),
            bg=self.header_bar_color,
            fg=self.text_color,
            anchor='w',
            padx=10,
            pady=8,
        )
        balanced_header.pack(fill=tk.X)

        balanced_body = tk.Frame(balanced_frame, bg=self.frame_color)
        balanced_body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.balanced_labels = {}
        for param in ['Speed', 'Altitude', 'Overlap rate', 'Scanning angle']:
            row = tk.Frame(balanced_body, bg=self.frame_color)
            row.pack(fill=tk.X, pady=3)
            name_label = tk.Label(
                row,
                text=f'{param}: ',
                bg=self.frame_color,
                fg=self.text_color,
                font=('Times New Roman', 13, 'bold'),
                anchor='w',
            )
            name_label.pack(side=tk.LEFT)
            value_label = tk.Label(
                row,
                text='--',
                bg=self.frame_color,
                fg=self.text_color,
                font=('Times New Roman', 13, 'bold'),
                anchor='w',
            )
            value_label.pack(side=tk.LEFT)
            self.balanced_labels[param] = value_label

        # ===== 下方：株高计算 =====
        self.bottom_card = tk.Frame(
            self.right_frame,
            bg=self.frame_color,
            bd=1,
            relief=tk.SOLID,
            highlightbackground=self.border_color,
            highlightthickness=1,
            padx=14,
            pady=12,
        )
        self.bottom_card.pack(fill=tk.X, pady=(12, 0))
        self.bottom_card.grid_columnconfigure(2, weight=1)

        tk.Label(
            self.bottom_card,
            text='Input',
            font=('Times New Roman', 13, 'bold'),
            bg=self.frame_color,
            fg=self.text_color,
        ).grid(row=0, column=0, rowspan=3, sticky='nw', padx=(0, 16), pady=(2, 0))

        lidar_height_btn = tk.Button(
            self.bottom_card,
            text='Lidar data',
            width=12,
            font=('Arial', 10, 'bold'),
            bg=self.button_bg,
            fg=self.text_color,
            activebackground=self.button_hover,
            relief=tk.GROOVE,
            bd=1,
            command=self.select_lidar_file_height,
        )
        lidar_height_btn.grid(row=0, column=1, sticky='w', padx=(0, 20), pady=(0, 8))

        vector_height_btn = tk.Button(
            self.bottom_card,
            text='Vector file',
            width=12,
            font=('Arial', 10, 'bold'),
            bg=self.button_bg,
            fg=self.text_color,
            activebackground=self.button_hover,
            relief=tk.GROOVE,
            bd=1,
            command=self.select_vector_file_height,
        )
        vector_height_btn.grid(row=1, column=1, sticky='w', padx=(0, 20), pady=(0, 8))

        start_height_btn = tk.Button(
            self.bottom_card,
            text='Start',
            width=12,
            font=('Times New Roman', 11, 'bold'),
            bg=self.button_bg,
            fg=self.value_color,
            activebackground=self.button_hover,
            relief=tk.GROOVE,
            bd=1,
            command=self.run_height_calculation,
        )
        start_height_btn.grid(row=2, column=1, sticky='w', padx=(0, 20))

        bottom_info = tk.Frame(self.bottom_card, bg=self.frame_color)
        bottom_info.grid(row=0, column=2, rowspan=3, sticky='nsew')
        bottom_info.grid_columnconfigure(1, weight=1)

        tk.Label(
            bottom_info,
            text='progress',
            font=('Times New Roman', 13, 'bold'),
            bg=self.frame_color,
            fg=self.text_color,
        ).grid(row=0, column=0, sticky='w', padx=(0, 12), pady=(0, 10))

        self.progress_bar2 = ttk.Progressbar(
            bottom_info,
            length=230,
            mode='determinate',
            style='green.Horizontal.TProgressbar',
        )
        self.progress_bar2.grid(row=0, column=1, sticky='w', pady=(0, 10))

        download_height_btn = tk.Button(
            bottom_info,
            text='Download Results',
            font=('Times New Roman', 13, 'bold'),
            width=16,
            height=1,
            bg=self.button_bg,
            fg=self.value_color,
            activebackground=self.button_hover,
            relief=tk.GROOVE,
            bd=1,
            command=self.download_height_results,
        )
        download_height_btn.grid(row=1, column=0, columnspan=2, sticky='w', pady=(0, 8))

        self.height_file_label = tk.Label(
            bottom_info,
            text='Result: waiting',
            bg=self.frame_color,
            fg=self.text_color,
            font=('Times New Roman', 13, 'bold'),
            anchor='w',
        )
        self.height_file_label.grid(row=2, column=0, columnspan=2, sticky='w')

        # 保留原逻辑所需标签，但不影响布局展示
        self.height_label = tk.Label(
            bottom_info,
            text='',
            bg=self.frame_color,
            fg=self.frame_color,
            font=('Times New Roman', 1),
        )
        self.height_label.grid(row=3, column=0, columnspan=2, sticky='w')

        self.create_left_panel(self.left_frame)
        self.root.after(150, self.align_left_panel_sections)
        self.root.after(450, self.align_left_panel_sections)
        self.root.bind('<Configure>', self._on_root_configure)



    # ------------------------------------------------------------------ #
    #  左侧面板
    # ------------------------------------------------------------------ #
    def create_left_panel(self, frame):
        items = [
            ('stage', 'Growth Stage\nDetection'),
            ('params', 'Optimal\nParameters'),
            ('height', 'Plant height\ncalculation'),
        ]

        self.left_nav_items = []
        self.left_nav_contents = []
        for _, (kind, text_value) in enumerate(items):
            section = tk.Frame(
                frame,
                bg=self.frame_color,
                bd=0,
                highlightthickness=0,
            )

            content = tk.Frame(section, bg=self.frame_color)

            icon_canvas = tk.Canvas(
                content,
                width=106,
                height=72,
                bg=self.frame_color,
                highlightthickness=0,
                bd=0,
            )
            if kind == 'height':
                icon_canvas.pack(pady=(0, 2))
            else:
                icon_canvas.pack(pady=(8, 6))
            self._draw_nav_icon(icon_canvas, kind)

            text_label = tk.Label(
                content,
                text=text_value,
                font=('Times New Roman', 15, 'bold'),
                bg=self.frame_color,
                fg=self.text_color,
                wraplength=124,
                justify=tk.CENTER,
            )
            if kind == 'height':
                text_label.pack(pady=(0, 0))
            else:
                text_label.pack()

            self.left_nav_items.append(section)
            self.left_nav_contents.append((section, content, kind))

    def _draw_nav_icon(self, canvas, kind):
        c = self.accent_color
        if kind == 'stage':
            # chain icon
            canvas.create_oval(16, 24, 50, 48, outline=c, width=6)
            canvas.create_oval(48, 24, 82, 48, outline=c, width=6)
            canvas.create_line(38, 36, 60, 36, fill=c, width=6, capstyle=tk.ROUND)
        elif kind == 'params':
            # gear icon
            canvas.create_oval(28, 12, 80, 64, outline=c, width=4)
            canvas.create_oval(45, 29, 63, 47, outline=c, width=3)
            for x1, y1, x2, y2 in [
                (54, 6, 54, 14), (54, 62, 54, 70),
                (22, 38, 30, 38), (78, 38, 86, 38),
                (34, 18, 28, 12), (74, 18, 80, 12),
                (34, 58, 28, 64), (74, 58, 80, 64),
            ]:
                canvas.create_line(x1, y1, x2, y2, fill=c, width=3, capstyle=tk.ROUND)
        elif kind == 'height':
            # tray / cabinet icon
            canvas.create_polygon(
                20, 52, 28, 24, 78, 24, 86, 52,
                outline=c, fill='', width=4, joinstyle=tk.ROUND
            )
            canvas.create_rectangle(20, 52, 86, 64, outline=c, width=4)
            canvas.create_line(32, 44, 74, 44, fill=c, width=3, capstyle=tk.ROUND)
            canvas.create_line(32, 58, 74, 58, fill=c, width=3, capstyle=tk.ROUND)

    def _on_root_configure(self, event=None):
        if event is None or event.widget == self.root:
            self.root.after_idle(self.align_left_panel_sections)

    def align_left_panel_sections(self):
        if not hasattr(self, 'left_nav_items') or len(self.left_nav_items) != 3:
            return
        self.root.update_idletasks()

        top_y = self.top_card.winfo_y()
        mid_y = self.params_frame.winfo_y()
        bottom_y = self.bottom_card.winfo_y()

        top_h = self.top_card.winfo_height()
        mid_h = self.params_frame.winfo_height()
        bottom_h = self.bottom_card.winfo_height()

        total_height = max((bottom_y + bottom_h) - top_y, 420)
        self.left_frame.configure(height=total_height)

        inner_x = 10
        inner_w = max(self.left_frame.winfo_width() - 20, 110)
        positions = [
            (0, top_h),
            (mid_y - top_y, mid_h),
            (bottom_y - top_y, bottom_h),
        ]

        for (section, (y_pos, sec_h)), (_, content, kind) in zip(zip(self.left_nav_items, positions), self.left_nav_contents):
            section.grid_forget()
            section.place(x=inner_x, y=y_pos, width=inner_w, height=sec_h)

            content.place_forget()
            if kind == 'height':
                # 第3个导航块单独上移
                content.place(relx=0.5, y=-10, anchor='n')
            else:
                content.place(relx=0.5, rely=0.5, anchor='center')

        self.left_bottom_line.place(x=0, y=total_height - 2, relwidth=1.0, height=1)
        self.left_bottom_line.lift()

    # ------------------------------------------------------------------ #
    #  文件选择
    # ------------------------------------------------------------------ #
    def select_lidar_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("LAS files", "*.las")])
        if file_path:
            self.lidar_file = file_path
            messagebox.showinfo("Success",
                                f"LiDAR file selected:\n{os.path.basename(file_path)}")

    def select_vector_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("SHP files", "*.shp")])
        if file_path:
            self.vector_file = file_path
            messagebox.showinfo("Success",
                                f"Vector file selected:\n{os.path.basename(file_path)}")

    def select_lidar_file_height(self):
        file_path = filedialog.askopenfilename(filetypes=[("LAS files", "*.las")])
        if file_path:
            self.lidar_file_height = file_path
            messagebox.showinfo("Success",
                                f"LiDAR file (height) selected:\n{os.path.basename(file_path)}")

    def select_vector_file_height(self):
        file_path = filedialog.askopenfilename(filetypes=[("SHP files", "*.shp")])
        if file_path:
            self.vector_file_height = file_path
            messagebox.showinfo("Success",
                                f"Vector file (height) selected:\n{os.path.basename(file_path)}")

    # ------------------------------------------------------------------ #
    #  运行分析
    # ------------------------------------------------------------------ #
    def run_stage_detection(self):
        if not self.lidar_file:
            messagebox.showerror("Error", "Please select a LiDAR file for stage detection")
            return
        if not self.vector_file:
            messagebox.showerror("Error", "Please select a vector file for stage detection")
            return
        if not os.path.exists(self.model_file):
            messagebox.showerror("Error", f"Model file not found:\n{self.model_file}")
            return
        if self.is_processing_stage:
            messagebox.showinfo("Info", "Stage detection is already running")
            return

        self.is_processing_stage = True
        self.progress_bar1['value'] = 0
        self.stage_label.config(text="Growth stage: processing...", fg=self.text_color)
        self.stage_file_label.config(text="Result: processing...", fg=self.text_color)
        for lbl in self.optimal_labels.values():
            lbl.config(text='--', fg=self.text_color)
        for lbl in self.balanced_labels.values():
            lbl.config(text='--', fg=self.text_color)

        thread = threading.Thread(target=self._run_stage_detection, daemon=True)
        thread.start()

    def _run_stage_detection(self):
        try:
            las_file = self.lidar_file
            shapefile_path = self.vector_file
            model_path = self.model_file

            # ==================== Step 1: SHP 分割 ====================
            self.root.after(0, lambda: self.progress_bar1.config(value=5))
            region_dir = self._divide_regions(las_file, shapefile_path)
            self.output_dir = region_dir

            self.root.after(0, lambda: self.progress_bar1.config(value=40))

            # ==================== Step 2: 生育期分类 ====================
            self.results = self._batch_classify_las(region_dir, model_path)

            self.root.after(0, lambda: self.progress_bar1.config(value=90))

            # 更新生育期 & 飞行参数
            stage = "Post-tasseling stage" if self.post_ratio > 50 else "Pre-tasseling stage"
            self.root.after(0,
                            lambda s=stage: self.stage_label.config(
                                text=f"Growth stage: {s}",
                                fg=self.value_color))
            self._update_flight_params()

            self.root.after(0, lambda: self.progress_bar1.config(value=100))

            # 保存结果
            saved_path = self._save_stage_results(las_file)

            self.root.after(
                0,
                lambda p=saved_path: messagebox.showinfo(
                    "Success",
                    f"Stage detection completed successfully!\n\nResults saved to:\n{p}" if p else
                    "Stage detection completed successfully!"
                ),
            )

        except Exception as e:
            self.root.after(0,
                            lambda: messagebox.showerror("Error",
                                                         f"Error during stage detection:\n{e}"))
        finally:
            self.is_processing_stage = False

    def run_height_calculation(self):
        if not self.lidar_file_height:
            messagebox.showerror("Error", "Please select a LiDAR file for height calculation")
            return
        if not self.vector_file_height:
            messagebox.showerror("Error", "Please select a vector file for height calculation")
            return
        if self.is_processing_height:
            messagebox.showinfo("Info", "Height calculation is already running")
            return

        self.is_processing_height = True
        self.progress_bar2['value'] = 0
        self.height_label.config(text="Height result: processing...")
        self.height_file_label.config(text="Result: processing...", fg=self.value_color)

        thread = threading.Thread(target=self._run_height_calculation, daemon=True)
        thread.start()

    def _run_height_calculation(self):
        try:
            las_file = self.lidar_file_height
            shapefile_path = self.vector_file_height

            # ==================== Step 1: SHP 分割 ====================
            self.root.after(0, lambda: self.progress_bar2.config(value=5))
            region_dir = self._divide_regions(las_file, shapefile_path)

            self.root.after(0, lambda: self.progress_bar2.config(value=30))

            # ==================== Step 2: 株高计算 ====================
            self.height_results = self._calculate_heights(region_dir)

            self.root.after(0, lambda: self.progress_bar2.config(value=100))

            if self.height_results:
                avg_h = np.mean([r['per_100'] for r in self.height_results
                                 if r['per_100'] is not None])
                self.root.after(0,
                                lambda: self.height_label.config(
                                    text=f"Height result: avg {avg_h:.2f} cm  "
                                         f"({len(self.height_results)} plots)"))

            # 保存结果
            saved_path = self._save_height_results(las_file)

            self.root.after(
                0,
                lambda p=saved_path: messagebox.showinfo(
                    "Success",
                    f"Height calculation completed successfully!\n\nResults saved to:\n{p}" if p else
                    "Height calculation completed successfully!"
                ),
            )

        except Exception as e:
            self.root.after(0,
                            lambda: messagebox.showerror("Error",
                                                         f"Error during height calculation:\n{e}"))
        finally:
            self.is_processing_height = False

    def _save_stage_results(self, las_file):
        output_folder = os.path.dirname(las_file)
        las_name = os.path.splitext(os.path.basename(las_file))[0]

        # 生育期分类结果
        if self.results:
            cls_file = os.path.join(output_folder,
                                    f"{las_name}_classification.txt")
            with open(cls_file, "w", encoding="utf-8") as f:
                f.write("PlotID,Classification\n")
                for file_name, result in self.results:
                    region_id = os.path.splitext(file_name)[0]
                    f.write(f"{region_id},{result}\n")
            self.last_stage_result_file = cls_file
            self.root.after(0, lambda p=cls_file: self.stage_file_label.config(
                text=f"Result: {os.path.basename(p)}", fg=self.value_color))
            print(f"[保存] 生育期结果: {cls_file}")
            return cls_file
        return ""

    def _save_height_results(self, las_file):
        output_folder = os.path.dirname(las_file)
        las_name = os.path.splitext(os.path.basename(las_file))[0]

        # 株高结果
        if self.height_results:
            df = pd.DataFrame(self.height_results)
            csv_file = os.path.join(output_folder, f"{las_name}_height.csv")
            df.to_csv(csv_file, index=False)
            self.last_height_result_file = csv_file
            self.root.after(0, lambda p=csv_file: self.height_file_label.config(
                text=f"Result: {os.path.basename(p)}", fg=self.value_color))
            print(f"[保存] 株高结果: {csv_file}")
            return csv_file
        return ""

    def download_stage_results(self):
        if not self.output_dir and not self.lidar_file:
            messagebox.showerror("Error",
                                 "No stage results to download. Please run stage detection first.")
            return
        target = self.output_dir if self.output_dir else os.path.dirname(
            self.lidar_file)
        os.startfile(target)
        messagebox.showinfo("Success", f"Stage results directory opened:\n{target}")

    def download_height_results(self):
        if not self.lidar_file_height:
            messagebox.showerror("Error",
                                 "No height results to download. Please run height calculation first.")
            return
        target = os.path.dirname(self.lidar_file_height)
        os.startfile(target)
        messagebox.showinfo("Success", f"Height results directory opened:\n{target}")

    # ================================================================== #
    #  1. SHP 分割（真实代码）
    # ================================================================== #
    def _divide_regions(self, las_file, shapefile_path, batch_size=10000000):
        print(f"[分割] 开始读取 LAS 文件: {las_file}")
        las = laspy.read(las_file)
        crs = las.header.parse_crs()
        print(f"[分割] 读取 LAS 完成，总点数: {len(las.x)}")

        print(f"[分割] 开始读取 SHP 文件: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        print(f"[分割] 读取 SHP 完成，小区数量: {len(gdf)}")

        input_name = os.path.splitext(os.path.basename(las_file))[0]
        output_dir = os.path.join(os.path.dirname(las_file), input_name)
        os.makedirs(output_dir, exist_ok=True)

        total_points = len(las.x)
        region_data = {region['Id']: {
            "points": [], "colors": [], "intensities": [],
            "classification": [], "polygon": region.geometry
        } for _, region in gdf.iterrows()}

        print("[分割] 开始逐批处理点云数据...")
        start_index = 0
        batch_number = 1
        while start_index < total_points:
            end_index = min(start_index + batch_size, total_points)
            batch_x = las.x[start_index:end_index]
            batch_y = las.y[start_index:end_index]
            batch_z = las.z[start_index:end_index]
            batch_red = las.red[start_index:end_index]
            batch_green = las.green[start_index:end_index]
            batch_blue = las.blue[start_index:end_index]
            batch_intensity = las.intensity[start_index:end_index]
            batch_classification = las.classification[start_index:end_index]

            batch_points = np.vstack([batch_x, batch_y, batch_z]).T
            batch_colors = np.vstack([batch_red, batch_green, batch_blue]).T

            for region_id, data in region_data.items():
                polygon = data["polygon"]
                bbox_mask = (
                    (batch_points[:, 0] >= polygon.bounds[0]) &
                    (batch_points[:, 0] <= polygon.bounds[2]) &
                    (batch_points[:, 1] >= polygon.bounds[1]) &
                    (batch_points[:, 1] <= polygon.bounds[3])
                )
                candidate_points = batch_points[bbox_mask]
                candidate_colors = batch_colors[bbox_mask]
                candidate_intensities = batch_intensity[bbox_mask]
                candidate_classification = batch_classification[bbox_mask]

                if len(candidate_points) > 0:
                    final_mask = np.array(
                        [polygon.contains(Point(p)) for p in candidate_points])
                    fp = candidate_points[final_mask]
                    fc = candidate_colors[final_mask]
                    fi = candidate_intensities[final_mask]
                    fcl = candidate_classification[final_mask]
                    data["points"].append(fp)
                    data["colors"].append(fc)
                    data["intensities"].append(fi)
                    data["classification"].append(fcl)

            print(f"[分割] 批次 {batch_number} 完成: "
                  f"点数 {end_index - start_index}")
            start_index = end_index
            batch_number += 1

        print("[分割] 保存小区点云文件...")
        for region_id, data in region_data.items():
            if len(data["points"]) == 0:
                print(f"[分割] 小区 {region_id}: 无点云，跳过")
                continue
            all_points = np.vstack(data["points"])
            if all_points.shape[0] == 0:
                continue
            all_colors = np.vstack(data["colors"])
            all_intensities = np.concatenate(data["intensities"])
            all_classification = np.concatenate(data["classification"])

            header = laspy.LasHeader(
                point_format=las.header.point_format,
                version=las.header.version)
            header.scale = las.header.scale
            header.offset = las.header.offset
            header.add_crs(crs)

            new_las = laspy.LasData(header)
            new_las.x = all_points[:, 0]
            new_las.y = all_points[:, 1]
            new_las.z = all_points[:, 2]
            new_las.red = all_colors[:, 0]
            new_las.green = all_colors[:, 1]
            new_las.blue = all_colors[:, 2]
            new_las.intensity = all_intensities
            new_las.classification = all_classification

            out_file = os.path.join(output_dir, f"{region_id}.las")
            new_las.write(out_file)
            print(f"[分割] 小区文件保存成功: {out_file}")

        print("[分割] 完成")
        return output_dir

    # ================================================================== #
    #  2. 生育期分类（真实代码）
    # ================================================================== #
    def _load_las_points(self, path, num_points=4096):
        las = laspy.read(path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        if len(points) >= num_points:
            idx = np.random.choice(len(points), num_points, replace=False)
        else:
            idx = np.random.choice(len(points), num_points, replace=True)
        points = points[idx, :].astype(np.float32)
        points -= np.mean(points, axis=0)
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if dist > 0:
            points /= dist
        return points

    def _batch_classify_las(self, folder_path, model_path):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--dropout', type=float, default=0.3)
        args = parser.parse_args([])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Pct(args, output_channels=2).to(device)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.eval()

        print(f"[分类] 模型加载成功，开始处理: {folder_path}")

        results = []
        post_count = 0
        total_count = 0
        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.las')]

        if not files:
            print("[分类] 文件夹中没有 .las 文件")
            return results

        with torch.no_grad():
            for file_name in files:
                file_path = os.path.join(folder_path, file_name)
                try:
                    points = self._load_las_points(file_path, num_points=4096)
                    data = torch.from_numpy(points).float().unsqueeze(0)
                    data = data.permute(0, 2, 1).to(device)
                    logits = model(data)
                    pred = logits.max(dim=1)[1].item()

                    label_map = {0: "Post-tasseling", 1: "Pre-tasseling"}
                    result_str = label_map.get(pred, f"Unknown({pred})")
                    print(f"[分类] {file_name} -> {result_str}")

                    results.append((file_name, result_str))
                    total_count += 1
                    if pred == 0:
                        post_count += 1
                except Exception as e:
                    print(f"[分类] 处理 {file_name} 出错: {e}")

        if total_count == 0:
            print("[分类] 没有有效数据")
            return results

        self.post_ratio = post_count / total_count * 100
        print(f"[分类] 总样本: {total_count}, 抽雄后: {post_count}, "
              f"比例: {self.post_ratio:.2f}%")
        return results

    # ================================================================== #
    #  3. 株高计算（真实代码，来自 calculate height2.1.py）
    # ================================================================== #
    def _classification(self, file_path):
        las = laspy.read(file_path)
        point_classification = las.classification
        input_folder = os.path.dirname(file_path)
        output_folder = os.path.join(input_folder, "classification")
        os.makedirs(output_folder, exist_ok=True)
        output_paths = {}

        for cls in [1, 2]:
            mask = point_classification == cls
            if not np.any(mask):
                continue
            las_class = laspy.create(
                point_format=las.header.point_format,
                file_version=las.header.version)
            las_class.points = las.points[mask]
            las_class.header.scale = las.header.scale
            las_class.header.offset = las.header.offset
            base_name = os.path.splitext(os.path.basename(file_path))[0].split('_')[0]
            output_path = os.path.join(output_folder,
                                       f"{base_name}_cls_{cls}.las")
            las_class.write(output_path)
            output_paths[cls] = output_path

        if 2 not in output_paths or 1 not in output_paths:
            return None, None

        ground_las = laspy.read(output_paths[2])
        non_ground_las = laspy.read(output_paths[1])
        return ground_las, non_ground_las

    def _vegetation_height_calculate(self, vegetation_las, ground_las, percen):
        ground_x = ground_las.x
        ground_y = ground_las.y
        ground_z = ground_las.z
        vegetation_x = vegetation_las.x
        vegetation_y = vegetation_las.y
        vegetation_z = vegetation_las.z

        min_x = max(ground_x.min(), vegetation_x.min())
        max_x = min(ground_x.max(), vegetation_x.max())
        min_y = max(ground_y.min(), vegetation_y.min())
        max_y = min(ground_y.max(), vegetation_y.max())

        border_margin = 0.2
        new_min_x = min_x + (max_x - min_x) * border_margin
        new_max_x = max_x - (max_x - min_x) * border_margin
        new_min_y = min_y + (max_y - min_y) * border_margin
        new_max_y = max_y - (max_y - min_y) * border_margin

        ground_mask = (
            (ground_x >= new_min_x) & (ground_x <= new_max_x) &
            (ground_y >= new_min_y) & (ground_y <= new_max_y))
        vegetation_mask = (
            (vegetation_x >= new_min_x) & (vegetation_x <= new_max_x) &
            (vegetation_y >= new_min_y) & (vegetation_y <= new_max_y))

        ground_x = ground_x[ground_mask]
        ground_y = ground_y[ground_mask]
        ground_z = ground_z[ground_mask]
        vegetation_x = vegetation_x[vegetation_mask]
        vegetation_y = vegetation_y[vegetation_mask]
        vegetation_z = vegetation_z[vegetation_mask]

        hight_gap = 0.03
        per = 0.04
        n = 0
        while True:
            gap_value = (np.percentile(vegetation_z, percen - n * per) -
                         np.percentile(vegetation_z, percen - (n + 1) * per))
            if gap_value < hight_gap:
                hight = np.percentile(vegetation_z, percen - n * per)
                break
            else:
                n += 1

        vegetation_coords = np.column_stack(
            (vegetation_x, vegetation_y, vegetation_z))
        tolerance = 1e-2
        matching_indices = np.where(np.abs(vegetation_z - hight) < tolerance)[0]
        same_height_coords = vegetation_coords[matching_indices]

        if len(same_height_coords) < 5:
            avg_ground_height = np.mean(ground_z)
            height = hight - avg_ground_height
            return height

        same_height_x = same_height_coords[:, 0]
        same_height_y = same_height_coords[:, 1]

        interpolated_dem = griddata(
            (ground_x, ground_y), ground_z,
            (same_height_x, same_height_y),
            method='cubic', fill_value=np.nan)
        interpolated_dsm = griddata(
            (vegetation_x, vegetation_y), vegetation_z,
            (same_height_x, same_height_y),
            method='nearest', fill_value=np.nan)
        interpolated_vegetation_height = interpolated_dsm - interpolated_dem

        ground_points = np.vstack([ground_x, ground_y, ground_z]).T
        ground_kd_tree = cKDTree(ground_points[:, :2])

        initial_radius = 0.03
        radius_step = 0.01
        max_radius = 0.2

        point_height = []
        valid_indices = []

        for idx, point in enumerate(same_height_coords):
            x, y, z = point
            radius = initial_radius
            while radius <= max_radius:
                indices = ground_kd_tree.query_ball_point([x, y], radius)
                if indices:
                    nearby_ground_points = ground_points[indices]
                    nearby_ground_heights = nearby_ground_points[:, 2]
                    if len(indices) > 2:
                        lower_p = np.percentile(nearby_ground_heights, 1)
                        upper_p = np.percentile(nearby_ground_heights, 99)
                        sorted_data = np.sort(nearby_ground_heights)
                        trimmed = sorted_data[
                            (sorted_data >= lower_p) & (sorted_data <= upper_p)]
                        avg_ground_height = np.mean(trimmed)
                    else:
                        avg_ground_height = np.mean(nearby_ground_heights)
                    height_diff_mean = z - avg_ground_height
                    point_height.append(height_diff_mean)
                    valid_indices.append(idx)
                    break
                else:
                    radius += radius_step

        if len(point_height) < 5:
            avg_ground_height = np.mean(ground_z)
            height = hight - avg_ground_height
            return height

        point_height = np.array(point_height)
        ivh_valid = interpolated_vegetation_height[valid_indices]
        differences_abs = np.abs(ivh_valid - point_height)

        min_five_indices = np.argsort(differences_abs)[:50]
        interp_heights = []
        diff_heights = []
        for index in min_five_indices:
            interp_heights.append(ivh_valid[index])
            diff_heights.append(point_height[index])

        if interp_heights and diff_heights:
            height = (max(interp_heights) + max(diff_heights)) / 2
        else:
            height = None
        return height

    def _calculate_heights(self, region_dir):
        las_paths = [os.path.join(region_dir, f)
                     for f in os.listdir(region_dir)
                     if f.lower().endswith('.las')]
        print(f"[株高] 共 {len(las_paths)} 个小区文件待处理")

        results = []
        total = len(las_paths)
        for i, file_path in enumerate(las_paths):
            try:
                ground_las, vegetation_las = self._classification(file_path)
                if ground_las is None or vegetation_las is None:
                    print(f"[株高] 小区 {os.path.basename(file_path)} 分类数据不足，跳过")
                    continue

                height_max = self._vegetation_height_calculate(
                    vegetation_las, ground_las, 100)
                height_1 = self._vegetation_height_calculate(
                    vegetation_las, ground_las, 99.5)
                height_2 = self._vegetation_height_calculate(
                    vegetation_las, ground_las, 99)

                if height_max is not None and height_1 is not None:
                    if (height_max < height_1) and (height_1 - height_max > 0.15):
                        height_max = height_1

                region_id = os.path.splitext(
                    os.path.basename(file_path))[0].split('_')[0]
                results.append({
                    'Id': region_id,
                    'per_100': round(height_max * 100, 2) if height_max else None,
                    'per_99.5': round(height_1 * 100, 2) if height_1 else None,
                    'per_99': round(height_2 * 100, 2) if height_2 else None,
                })
                print(f"[株高] {region_id}: "
                      f"per_100={results[-1]['per_100']}  "
                      f"per_99.5={results[-1]['per_99.5']}  "
                      f"per_99={results[-1]['per_99']}")

                progress = int(30 + 70 * (i + 1) / total)
                self.root.after(0,
                                lambda v=progress: self.progress_bar2.config(value=v))

            except Exception as e:
                print(f"[株高] 处理 {os.path.basename(file_path)} 出错: {e}")

        return results

    # ================================================================== #
    #  界面更新 & 保存
    # ================================================================== #
    def _update_flight_params(self):
        if self.post_ratio > 50:
            opt = {"Speed": "2 m/s", "Altitude": "25 m",
                   "Overlap rate": "50%", "Scanning angle": "45°"}
            bal = {"Speed": "4 m/s", "Altitude": "25 m",
                   "Overlap rate": "30%", "Scanning angle": "45°"}
        else:
            opt = {"Speed": "2 m/s", "Altitude": "25 m",
                   "Overlap rate": "30%", "Scanning angle": "90°"}
            bal = {"Speed": "4 m/s", "Altitude": "25 m",
                   "Overlap rate": "30%", "Scanning angle": "90°"}

        for k, v in opt.items():
            self.root.after(0,
                            lambda k=k, v=v: self.optimal_labels[k].config(
                                text=v,
                                fg=self.value_color))
        for k, v in bal.items():
            self.root.after(0,
                            lambda k=k, v=v: self.balanced_labels[k].config(
                                text=v,
                                fg=self.value_color))


if __name__ == "__main__":
    root = tk.Tk()
    app = LiDAROptimizerApp(root)
    root.mainloop()