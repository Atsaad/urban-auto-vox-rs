#!/usr/bin/env python3
"""
Modern Voxel Pipeline GUI (v3)
CustomTkinter interface with redesigned layout, archive support, and verbose terminal

Layout:
  Left panel  — Config sections + action buttons (no scrolling needed)
  Right panel — Compact status row + unified terminal (log + docker compose)
"""

import customtkinter as ctk
from tkinter import messagebox, scrolledtext, filedialog
import tkinter as tk
import subprocess
import threading
import time
import os
import re
from datetime import datetime

# ============================================================================
# THEME CONFIGURATION
# ============================================================================

DARK_COLORS = {
    'primary': '#3498db',
    'primary_hover': '#2980b9',
    'success': '#27ae60',
    'success_hover': '#2ecc71',
    'warning': '#f39c12',
    'warning_hover': '#e67e22',
    'danger': '#e74c3c',
    'danger_hover': '#c0392b',
    'dark': '#1a1a1a',
    'darker': '#111111',
    'card': '#2a2a2a',
    'card_hover': '#333333',
    'accent': '#3a3a3a',
    'text': '#ecf0f1',
    'text_muted': '#888888',
    'border': '#444444',
    'log_bg': '#111111',
    'log_fg': '#e0e0e0',
}

LIGHT_COLORS = {
    'primary': '#3498db',
    'primary_hover': '#2980b9',
    'success': '#27ae60',
    'success_hover': '#2ecc71',
    'warning': '#f39c12',
    'warning_hover': '#e67e22',
    'danger': '#e74c3c',
    'danger_hover': '#c0392b',
    'dark': '#f5f5f5',
    'darker': '#ffffff',
    'card': '#ffffff',
    'card_hover': '#f0f0f0',
    'accent': '#e8e8e8',
    'text': '#212529',
    'text_muted': '#6c757d',
    'border': '#dee2e6',
    'log_bg': '#fafafa',
    'log_fg': '#212529',
}

COLORS = dict(DARK_COLORS)

def theme(key):
    """Return (light, dark) color tuple for CTk widgets"""
    return (LIGHT_COLORS[key], DARK_COLORS[key])

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ModernVoxelGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🏗️ AutoVox Pipeline Manager — Rust edition")
        self.geometry("1500x950")
        self.minsize(1200, 800)

        self.working_dir = os.path.dirname(os.path.abspath(__file__))
        self.docker_compose_path = os.path.join(self.working_dir, "docker-compose.yml")

        # Configuration variables
        self.citygml_version = ctk.StringVar(value="2.0")
        self.opt_tbw = ctk.BooleanVar(value=True)
        self.opt_add_bb = ctk.BooleanVar(value=False)
        self.opt_add_json = ctk.BooleanVar(value=True)
        self.opt_group_sc = ctk.BooleanVar(value=False)
        self.opt_group_scomp = ctk.BooleanVar(value=False)

        self.db_host_port = ctk.StringVar(value="5434")
        self.db_name = ctk.StringVar(value="voxel_db")
        self.db_user = ctk.StringVar(value="postgres")
        self.db_password = ctk.StringVar(value="postgres")
        self.db_srid = ctk.StringVar(value="25832")

        self.voxel_size = ctk.StringVar(value="0.5")
        self.num_workers = ctk.IntVar(value=8)

        self.output_format = ctk.StringVar(value="csv")
        self.processing_mode = ctk.StringVar(value="single")

        self.batch_source_dir = ctk.StringVar(value="")
        self.batch_max_batches = ctk.StringVar(value="0")
        self.batch_process_all = ctk.BooleanVar(value=True)
        self.batch_auto_zip = ctk.BooleanVar(value=True)
        self.batch_output_dir = ctk.StringVar(value="./output_batches")

        # Three boxes mirror the actual container graph in this workspace's
        # docker-compose.yml. The Rust voxelizer auto-derives translate.json,
        # index.json, and grid_mapping.json inside its own container, so the
        # old "Translate" / "Index" steps are folded into "Voxelize+Ingest".
        self.pipeline_steps = [
            {"name": "CityGML 2→3",     "status": "pending", "icon": "📄"},
            {"name": "OBJ + JSON",      "status": "pending", "icon": "🔄"},
            {"name": "Voxelize+Ingest", "status": "pending", "icon": "🧊"},
        ]
        self.current_step = -1
        self.refresh_timer = None
        self.auto_refresh = ctk.BooleanVar(value=True)
        self._command_running = False
        self._last_output_time = 0.0

        self.create_ui()
        self.refresh_status()
        self.start_auto_refresh()

        self.log_message("🚀 AutoVox Pipeline Manager — Rust edition", "info")
        self.log_message(f"📁 Working directory: {self.working_dir}", "info")
        self.log_message("   ./start.sh (single tile) · ./batch-process.sh (many tiles)", "info")
        self.log_message("✨ Configure settings → Write Config → Start Pipeline\n", "success")

    # ========================================================================
    # UI LAYOUT
    # ========================================================================

    def create_ui(self):
        """Build the main user interface — Left: config, Right: status + terminal"""
        self.grid_columnconfigure(0, weight=3)   # Left 60%
        self.grid_columnconfigure(1, weight=2)   # Right 40%
        self.grid_rowconfigure(0, weight=1)

        # ---- LEFT PANEL ----
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)  # config area expands

        self.create_header(left)                 # row 0
        self.create_pipeline_steps(left)         # row 1

        config = ctk.CTkScrollableFrame(left, fg_color="transparent")
        config.grid(row=2, column=0, sticky="nsew", pady=5)
        config.grid_columnconfigure(0, weight=1)

        self.create_citygml_section(config)
        self.create_rustgml_section(config)
        self.create_database_section(config)
        self.create_voxelizer_section(config)
        self.create_output_section(config)
        self.create_batch_section(config)

        self.create_action_buttons(left)         # row 3

        # ---- RIGHT PANEL ----
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)     # terminal expands

        self.create_status_row(right)            # row 0
        self.create_terminal_panel(right)         # row 1

    # ========================================================================
    # HEADER (compact)
    # ========================================================================

    def create_header(self, parent):
        header = ctk.CTkFrame(parent, fg_color=theme('card'), corner_radius=15)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        inner = ctk.CTkFrame(header, fg_color="transparent")
        inner.pack(fill="x", padx=15, pady=12)

        ctk.CTkLabel(
            inner, text="🏗️ AutoVox Pipeline",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(side="left")

        # Theme toggle
        self.theme_switch = ctk.CTkSwitch(
            inner, text="🌙", command=self.toggle_theme,
            onvalue="dark", offvalue="light", width=40
        )
        self.theme_switch.select()
        self.theme_switch.pack(side="right", padx=5)

        # Processing mode toggle
        self.mode_toggle = ctk.CTkSegmentedButton(
            inner, values=["🔹 Single", "📦 Batch"],
            command=self._on_mode_change,
            font=ctk.CTkFont(size=11, weight="bold"),
            selected_color=DARK_COLORS['primary'],
            selected_hover_color=DARK_COLORS['primary_hover'],
        )
        self.mode_toggle.set("🔹 Single")
        self.mode_toggle.pack(side="right", padx=10)

    # ========================================================================
    # PIPELINE STEPS
    # ========================================================================

    def create_pipeline_steps(self, parent):
        steps_frame = ctk.CTkFrame(parent, fg_color=theme('card'), corner_radius=15)
        steps_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))

        container = ctk.CTkFrame(steps_frame, fg_color="transparent")
        container.pack(fill="x", padx=15, pady=10)

        self.step_widgets = []
        for i, step in enumerate(self.pipeline_steps):
            sf = ctk.CTkFrame(container, fg_color=theme('accent'), corner_radius=10)
            sf.pack(side="left", expand=True, fill="x", padx=3)

            icon = ctk.CTkLabel(sf, text=step["icon"], font=ctk.CTkFont(size=18))
            icon.pack(pady=(6, 2))
            name = ctk.CTkLabel(sf, text=step["name"], font=ctk.CTkFont(size=10, weight="bold"))
            name.pack()
            status = ctk.CTkLabel(sf, text="○", font=ctk.CTkFont(size=9), text_color=theme('text_muted'))
            status.pack(pady=(0, 6))

            self.step_widgets.append({"frame": sf, "icon": icon, "name": name, "status": status})

            if i < len(self.pipeline_steps) - 1:
                ctk.CTkLabel(container, text="→", font=ctk.CTkFont(size=16, weight="bold"),
                             text_color=theme('text_muted')).pack(side="left", padx=1)

    def update_step_status(self, step_index, status):
        if step_index < 0 or step_index >= len(self.step_widgets):
            return
        w = self.step_widgets[step_index]
        if status == "running":
            w["frame"].configure(fg_color=theme('warning'))
            w["status"].configure(text="⏳", text_color=COLORS['dark'])
        elif status == "complete":
            w["frame"].configure(fg_color=theme('success'))
            w["status"].configure(text="✓", text_color="white")
        elif status == "error":
            w["frame"].configure(fg_color=theme('danger'))
            w["status"].configure(text="✗", text_color="white")
        else:
            w["frame"].configure(fg_color=theme('accent'))
            w["status"].configure(text="○", text_color=theme('text_muted'))

    # ========================================================================
    # CONFIG SECTIONS
    # ========================================================================

    def create_citygml_section(self, parent):
        section = ctk.CTkFrame(parent, fg_color=theme('card'), corner_radius=12)
        section.pack(fill="x", pady=3, padx=3)
        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 5))
        ctk.CTkLabel(header, text="1 CityGML", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        content = ctk.CTkFrame(section, fg_color="transparent")
        content.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkRadioButton(content, text="2.0 (needs upgrade)", variable=self.citygml_version, value="2.0").pack(side="left", padx=(0, 15))
        ctk.CTkRadioButton(content, text="3.0 (skip upgrade)", variable=self.citygml_version, value="3.0").pack(side="left")

    def create_rustgml_section(self, parent):
        section = ctk.CTkFrame(parent, fg_color=theme('card'), corner_radius=12)
        section.pack(fill="x", pady=3, padx=3)
        ctk.CTkLabel(section, text="2 RustGML2OBJ", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=12, pady=(10, 5))
        content = ctk.CTkFrame(section, fg_color="transparent")
        content.pack(fill="x", padx=12, pady=(0, 10))
        r1 = ctk.CTkFrame(content, fg_color="transparent")
        r1.pack(fill="x", pady=1)
        ctk.CTkCheckBox(r1, text="--tbw", variable=self.opt_tbw).pack(side="left", padx=(0, 15))
        ctk.CTkCheckBox(r1, text="--add-bb", variable=self.opt_add_bb).pack(side="left", padx=15)
        ctk.CTkCheckBox(r1, text="--add-json", variable=self.opt_add_json).pack(side="left", padx=15)
        ctk.CTkCheckBox(r1, text="--group-sc", variable=self.opt_group_sc).pack(side="left", padx=15)

    def create_database_section(self, parent):
        section = ctk.CTkFrame(parent, fg_color=theme('card'), corner_radius=12)
        section.pack(fill="x", pady=3, padx=3)
        ctk.CTkLabel(section, text="3 Database", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=12, pady=(10, 5))
        content = ctk.CTkFrame(section, fg_color="transparent")
        content.pack(fill="x", padx=12, pady=(0, 10))
        content.grid_columnconfigure((1, 3, 5), weight=1)
        # Row 1: Host Port (external) / SRID
        ctk.CTkLabel(content, text="Host port:").grid(row=0, column=0, sticky="e", padx=3, pady=3)
        ctk.CTkEntry(content, textvariable=self.db_host_port, width=60).grid(row=0, column=1, sticky="w", padx=3, pady=3)
        ctk.CTkLabel(content, text="SRID:").grid(row=0, column=2, sticky="e", padx=3, pady=3)
        ctk.CTkEntry(content, textvariable=self.db_srid, width=60).grid(row=0, column=3, sticky="w", padx=3, pady=3)
        # Row 2: DB / User / Pass
        ctk.CTkLabel(content, text="DB:").grid(row=1, column=0, sticky="e", padx=3, pady=3)
        ctk.CTkEntry(content, textvariable=self.db_name, width=120).grid(row=1, column=1, sticky="w", padx=3, pady=3)
        ctk.CTkLabel(content, text="User:").grid(row=1, column=2, sticky="e", padx=3, pady=3)
        ctk.CTkEntry(content, textvariable=self.db_user, width=60).grid(row=1, column=3, sticky="w", padx=3, pady=3)
        ctk.CTkLabel(content, text="Pass:").grid(row=1, column=4, sticky="e", padx=3, pady=3)
        ctk.CTkEntry(content, textvariable=self.db_password, width=60, show="*").grid(row=1, column=5, sticky="w", padx=3, pady=3)

    def create_voxelizer_section(self, parent):
        section = ctk.CTkFrame(parent, fg_color=theme('card'), corner_radius=12)
        section.pack(fill="x", pady=3, padx=3)
        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 5))
        ctk.CTkLabel(header, text="4 Voxelizer", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        ctk.CTkLabel(header, text="v7.4", font=ctk.CTkFont(size=9, weight="bold"),
                     fg_color=theme('success'), corner_radius=4, padx=6, pady=1).pack(side="left", padx=8)
        content = ctk.CTkFrame(section, fg_color="transparent")
        content.pack(fill="x", padx=12, pady=(0, 10))
        row = ctk.CTkFrame(content, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkLabel(row, text="Voxel size:").pack(side="left", padx=(0, 5))
        ctk.CTkOptionMenu(row, variable=self.voxel_size, values=["0.25", "0.5", "1.0", "2.0"],
                          width=80, fg_color=theme('primary'), button_color=theme('primary_hover')).pack(side="left", padx=5)
        ctk.CTkLabel(row, text="m").pack(side="left")
        ctk.CTkLabel(row, text="  Workers:").pack(side="left", padx=(15, 5))
        ws = ctk.CTkSlider(row, from_=1, to=32, number_of_steps=31, variable=self.num_workers, width=120)
        ws.pack(side="left", padx=5)
        self.workers_label = ctk.CTkLabel(row, text="8")
        self.workers_label.pack(side="left")
        def upd(v):
            self.workers_label.configure(text=str(int(float(v))))
            self.num_workers.set(int(float(v)))
        ws.configure(command=upd)

    def create_output_section(self, parent):
        self.output_section_frame = ctk.CTkFrame(parent, fg_color=theme('card'), corner_radius=12)
        self.output_section_frame.pack(fill="x", pady=3, padx=3)
        section = self.output_section_frame
        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 5))
        ctk.CTkLabel(header, text="5 Output", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        ctk.CTkLabel(header, text="v7.4", font=ctk.CTkFont(size=9, weight="bold"),
                     fg_color=theme('primary'), corner_radius=4, padx=6, pady=1).pack(side="left", padx=8)
        content = ctk.CTkFrame(section, fg_color="transparent")
        content.pack(fill="x", padx=12, pady=(0, 10))
        row = ctk.CTkFrame(content, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkRadioButton(row, text="CSV", variable=self.output_format, value="csv",
                           command=self._on_output_format_change).pack(side="left", padx=(0, 12))
        ctk.CTkRadioButton(row, text="PostGIS", variable=self.output_format, value="postgis",
                           command=self._on_output_format_change).pack(side="left", padx=12)
        ctk.CTkRadioButton(row, text="CSV + PostGIS Both", variable=self.output_format, value="both",
                           command=self._on_output_format_change).pack(side="left", padx=12)
        self.output_desc = ctk.CTkLabel(content, text="CSV: portable backup file",
                                        text_color=theme('text_muted'), font=ctk.CTkFont(size=10))
        self.output_desc.pack(anchor="w", pady=(4, 0))

    def _on_output_format_change(self):
        descs = {"csv": "CSV: portable backup file", "postgis": "PostGIS: live spatial queries (DB must be running)",
                 "both": "Both: CSV backup + live PostGIS"}
        self.output_desc.configure(text=descs.get(self.output_format.get(), ""))

    def _on_mode_change(self, value):
        if "Single" in value:
            self.processing_mode.set("single")
            self.batch_section.pack_forget()
            self.start_btn.configure(text="▶ Start Pipeline")
            self.log_message("🔹 Single Processing mode", "info")
        else:
            self.processing_mode.set("batch")
            self.batch_section.pack(fill="x", pady=3, padx=3, after=self.output_section_frame)
            self.start_btn.configure(text="📦 Start Batch")
            self.log_message("📦 Batch Processing mode", "info")
            self._update_tile_count()

    # ========================================================================
    # BATCH SECTION
    # ========================================================================

    def create_batch_section(self, parent):
        self.batch_section = ctk.CTkFrame(parent, fg_color=theme('card'), corner_radius=12)
        header = ctk.CTkFrame(self.batch_section, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 5))
        ctk.CTkLabel(header, text="6️⃣ Batch", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        ctk.CTkLabel(header, text="v3", font=ctk.CTkFont(size=9, weight="bold"),
                     fg_color=theme('warning'), corner_radius=4, padx=6, pady=1).pack(side="left", padx=8)

        content = ctk.CTkFrame(self.batch_section, fg_color="transparent")
        content.pack(fill="x", padx=12, pady=(0, 10))
        content.grid_columnconfigure(1, weight=1)

        # Source dir
        ctk.CTkLabel(content, text="Source:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=3, pady=5)
        src_row = ctk.CTkFrame(content, fg_color="transparent")
        src_row.grid(row=0, column=1, sticky="ew", padx=3, pady=5)
        src_row.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(src_row, textvariable=self.batch_source_dir, placeholder_text="Folder with tile subfolders or loose .gml files...").grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ctk.CTkButton(src_row, text="📂", width=35, command=self._browse_source_dir, fg_color=theme('accent')).grid(row=0, column=1)

        self.tile_count_label = ctk.CTkLabel(content, text="  No directory selected",
                                              text_color=theme('text_muted'), font=ctk.CTkFont(size=10))
        self.tile_count_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=3)

        # Max batches
        ctk.CTkLabel(content, text="Max:", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, sticky="w", padx=3, pady=3)
        max_row = ctk.CTkFrame(content, fg_color="transparent")
        max_row.grid(row=2, column=1, sticky="ew", padx=3, pady=3)
        self.batch_max_entry = ctk.CTkEntry(max_row, textvariable=self.batch_max_batches, width=60, state="disabled")
        self.batch_max_entry.pack(side="left", padx=(0, 8))
        ctk.CTkCheckBox(max_row, text="All", variable=self.batch_process_all, command=self._toggle_max_batches).pack(side="left")
        ctk.CTkSwitch(max_row, text="Zip", variable=self.batch_auto_zip).pack(side="left", padx=15)

        # Output dir
        ctk.CTkLabel(content, text="Output:", font=ctk.CTkFont(weight="bold")).grid(row=3, column=0, sticky="w", padx=3, pady=5)
        out_row = ctk.CTkFrame(content, fg_color="transparent")
        out_row.grid(row=3, column=1, sticky="ew", padx=3, pady=5)
        out_row.grid_columnconfigure(0, weight=1)
        ctk.CTkEntry(out_row, textvariable=self.batch_output_dir).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ctk.CTkButton(out_row, text="📂", width=35, command=self._browse_output_dir, fg_color=theme('accent')).grid(row=0, column=1)

        ctk.CTkLabel(content, text="ℹ️ Supports folders + loose .gml files. Resume-capable. Retry on failure.",
                     text_color=theme('text_muted'), font=ctk.CTkFont(size=10), wraplength=450, justify="left"
                     ).grid(row=4, column=0, columnspan=2, sticky="w", padx=3, pady=(5, 0))

    def _browse_source_dir(self):
        path = filedialog.askdirectory(title="Select Tiles Source Directory")
        if path:
            self.batch_source_dir.set(path)
            self._update_tile_count()

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select Batch Output Directory")
        if path:
            self.batch_output_dir.set(path)

    def _toggle_max_batches(self):
        if self.batch_process_all.get():
            self.batch_max_entry.configure(state="disabled")
            self.batch_max_batches.set("0")
        else:
            self.batch_max_entry.configure(state="normal")
            self.batch_max_batches.set("10")

    def _update_tile_count(self):
        src = self.batch_source_dir.get()
        if not src or not os.path.isdir(src):
            self.tile_count_label.configure(text="  ⚠️ Directory not found")
            return
        try:
            folders = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
            loose = [f for f in os.listdir(src) if f.endswith('.gml') and os.path.isfile(os.path.join(src, f))]
            gml = len(loose)
            for tf in folders:
                gml += sum(1 for f in os.listdir(os.path.join(src, tf)) if f.endswith('.gml'))
            total = len(folders) + len(loose)
            parts = []
            if folders:
                parts.append(f"{len(folders)} folders")
            if loose:
                parts.append(f"{len(loose)} loose files")
            self.tile_count_label.configure(
                text=f"  📊 {total} tiles ({', '.join(parts)}, {gml} GML total)",
                text_color=COLORS['success'])
        except Exception as e:
            self.tile_count_label.configure(text=f"  ❌ {e}")

    # ========================================================================
    # ACTION BUTTONS
    # ========================================================================

    def create_action_buttons(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=theme('card'), corner_radius=12)
        frame.grid(row=3, column=0, sticky="ew", pady=5)
        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.pack(fill="x", padx=10, pady=10)

        # Row 1: Primary actions
        r1 = ctk.CTkFrame(inner, fg_color="transparent")
        r1.pack(fill="x", pady=3)
        r1.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkButton(r1, text="💾 Write Config", command=self.write_configuration,
                      fg_color=theme('primary'), hover_color=theme('primary_hover'),
                      height=36, font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, padx=3, sticky="ew")

        self.start_btn = ctk.CTkButton(r1, text="▶ Start Pipeline", command=self.start_pipeline,
                                        fg_color=theme('success'), hover_color=theme('success_hover'),
                                        height=36, font=ctk.CTkFont(size=12, weight="bold"))
        self.start_btn.grid(row=0, column=1, padx=3, sticky="ew")

        ctk.CTkButton(r1, text="📦 Archive", command=self.archive_results,
                      fg_color=theme('warning'), hover_color=theme('warning_hover'),
                      height=36, font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=2, padx=3, sticky="ew")

        ctk.CTkButton(r1, text="🗑 Clean Data", command=self.clean_data,
                      fg_color=theme('accent'), hover_color=theme('card_hover'),
                      height=36).grid(row=0, column=3, padx=3, sticky="ew")

        # Row 2: Secondary actions
        r2 = ctk.CTkFrame(inner, fg_color="transparent")
        r2.pack(fill="x", pady=3)
        r2.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkButton(r2, text="🗄️ Clear DB", command=self.clear_database,
                      fg_color=theme('accent'), hover_color=theme('card_hover'), height=30).grid(row=0, column=0, padx=3, sticky="ew")
        ctk.CTkButton(r2, text="🔄 Restart DB", command=self.restart_database,
                      fg_color=theme('accent'), hover_color=theme('card_hover'), height=30).grid(row=0, column=1, padx=3, sticky="ew")
        ctk.CTkButton(r2, text="🐳 Clean Docker", command=self.clean_docker,
                      fg_color=theme('danger'), hover_color=theme('danger_hover'), height=30).grid(row=0, column=2, padx=3, sticky="ew")
        ctk.CTkButton(r2, text="🐳 View Compose", command=self.view_docker_compose,
                      fg_color=theme('accent'), hover_color=theme('card_hover'), height=30).grid(row=0, column=3, padx=3, sticky="ew")

    # ========================================================================
    # STATUS ROW (compact)
    # ========================================================================

    def create_status_row(self, parent):
        row = ctk.CTkFrame(parent, fg_color=theme('card'), corner_radius=12)
        row.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        row.grid_columnconfigure((0, 1, 2), weight=1)

        self.docker_status = ctk.CTkLabel(row, text="🐳 …", font=ctk.CTkFont(size=11),
                                           text_color=theme('text_muted'))
        self.docker_status.grid(row=0, column=0, padx=10, pady=8, sticky="w")

        self.files_status = ctk.CTkLabel(row, text="📁 …", font=ctk.CTkFont(size=11),
                                          text_color=theme('text_muted'))
        self.files_status.grid(row=0, column=1, padx=10, pady=8)

        self.db_status = ctk.CTkLabel(row, text="🗄️ …", font=ctk.CTkFont(size=11),
                                       text_color=theme('text_muted'))
        self.db_status.grid(row=0, column=2, padx=10, pady=8, sticky="e")

    # ========================================================================
    # TERMINAL PANEL (unified log + docker compose view)
    # ========================================================================

    def create_terminal_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=theme('card'), corner_radius=12)
        frame.grid(row=1, column=0, sticky="nsew", pady=(0, 0))
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 3))
        ctk.CTkLabel(header, text="📋 Terminal", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        ctk.CTkButton(header, text="Clear", command=self.clear_terminal, width=50, height=22,
                      fg_color=theme('accent'), hover_color=theme('card_hover')).pack(side="right", padx=2)

        self.terminal_text = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, font=('Courier', 10),
            bg=COLORS['log_bg'], fg=COLORS['log_fg'],
            insertbackground='white', relief='flat', padx=10, pady=8
        )
        self.terminal_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self.terminal_text.tag_config("success", foreground="#27ae60")
        self.terminal_text.tag_config("error", foreground="#e74c3c")
        self.terminal_text.tag_config("warning", foreground="#f39c12")
        self.terminal_text.tag_config("info", foreground="#3498db")
        self.terminal_text.tag_config("timestamp", foreground="#7f8c8d")
        self.terminal_text.tag_config("docker", foreground="#9b59b6")

    # ========================================================================
    # TERMINAL HELPERS
    # ========================================================================

    def log_message(self, message, tag=""):
        ts = datetime.now().strftime("[%H:%M:%S]")
        self.terminal_text.config(state='normal')
        self.terminal_text.insert(tk.END, ts + " ", "timestamp")
        self.terminal_text.insert(tk.END, message + "\n", tag)
        self.terminal_text.see(tk.END)
        self.terminal_text.config(state='disabled')

    def clear_terminal(self):
        self.terminal_text.config(state='normal')
        self.terminal_text.delete(1.0, tk.END)
        self.terminal_text.config(state='disabled')

    def view_docker_compose(self):
        """Display docker-compose.yml content in the terminal"""
        try:
            with open(self.docker_compose_path, 'r') as f:
                content = f.read()
            self.log_message("─" * 50, "info")
            self.log_message("📄 docker-compose.yml:", "info")
            self.log_message("─" * 50, "info")
            self.terminal_text.config(state='normal')
            self.terminal_text.insert(tk.END, content + "\n")
            self.terminal_text.see(tk.END)
            self.terminal_text.config(state='disabled')
            self.log_message("─" * 50, "info")
        except Exception as e:
            self.log_message(f"❌ Error reading compose file: {e}", "error")

    # ========================================================================
    # THEME
    # ========================================================================

    def toggle_theme(self):
        global COLORS
        current = ctk.get_appearance_mode()
        if current == "Dark":
            ctk.set_appearance_mode("light")
            self.theme_switch.configure(text="☀️")
            COLORS.update(LIGHT_COLORS)
        else:
            ctk.set_appearance_mode("dark")
            self.theme_switch.configure(text="🌙")
            COLORS.update(DARK_COLORS)
        self.apply_theme()

    def apply_theme(self):
        self.terminal_text.config(bg=COLORS['log_bg'], fg=COLORS['log_fg'])
        self.terminal_text.tag_config("success", foreground=COLORS['success'])
        self.terminal_text.tag_config("error", foreground=COLORS['danger'])
        self.terminal_text.tag_config("warning", foreground=COLORS['warning'])
        self.terminal_text.tag_config("info", foreground=COLORS['primary'])
        self.terminal_text.tag_config("timestamp", foreground=COLORS['text_muted'])

    # ========================================================================
    # AUTO-REFRESH
    # ========================================================================

    def toggle_auto_refresh(self):
        if self.auto_refresh.get():
            self.start_auto_refresh()
        else:
            self.stop_auto_refresh()

    def start_auto_refresh(self):
        self.refresh_status()
        self.refresh_timer = self.after(5000, self.start_auto_refresh)

    def stop_auto_refresh(self):
        if self.refresh_timer:
            self.after_cancel(self.refresh_timer)
            self.refresh_timer = None

    def refresh_status(self):
        """Refresh the compact status row"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '-a', '--format', '{{.Names}}: {{.Status}}'],
                capture_output=True, text=True, timeout=5
            )
            docker_lines = [l for l in result.stdout.strip().split('\n') if 'voxel' in l.lower()]
            running = sum(1 for l in docker_lines if 'Up' in l)
            total = len(docker_lines)
            self.docker_status.configure(
                text=f"🐳 {running}/{total} up" if total > 0 else "🐳 no containers",
                text_color=COLORS['success'] if running > 0 else COLORS['text_muted']
            )
        except Exception:
            self.docker_status.configure(text="🐳 error", text_color=COLORS['danger'])

        try:
            c2 = len([f for f in os.listdir(os.path.join(self.working_dir, 'data/citygml2')) if f.endswith('.gml')])
            objs = len([f for f in os.listdir(os.path.join(self.working_dir, 'data/objs')) if f.endswith('.obj') or f.endswith('.json')])
            self.files_status.configure(text=f"📁 In:{c2} Out:{objs}")
        except Exception:
            self.files_status.configure(text="📁 —")

        try:
            db_r = subprocess.run(
                ['docker', 'ps', '--filter', 'name=voxel_postgis', '--format', '{{.Status}}'],
                capture_output=True, text=True, timeout=5
            )
            if 'Up' in db_r.stdout:
                self.db_status.configure(text="🗄️ connected", text_color=COLORS['success'])
            else:
                self.db_status.configure(text="🗄️ offline", text_color=COLORS['text_muted'])
        except Exception:
            self.db_status.configure(text="🗄️ error", text_color=COLORS['danger'])

    # ========================================================================
    # WRITE CONFIGURATION
    # ========================================================================

    def write_configuration(self):
        try:
            self.log_message("─" * 40, "info")
            self.log_message("📝 Writing configuration to .env...", "info")
            env_path = os.path.join(self.working_dir, ".env")
            # Build rustgml2obj flags from checkboxes
            gml2obj_flags = []
            if self.opt_tbw.get():
                gml2obj_flags.append('--tbw')
            if self.opt_add_bb.get():
                gml2obj_flags.append('--add-bb')
            if self.opt_add_json.get():
                gml2obj_flags.append('--add-json')
            if self.opt_group_sc.get():
                gml2obj_flags.append('--group-sc')
            if self.opt_group_scomp.get():
                gml2obj_flags.append('--group-scomp')

            lines = [
                "# Auto-generated by AutoVox GUI",
                f"CITYGML_INPUT_VERSION={self.citygml_version.get()}",
                f"POSTGRES_DB={self.db_name.get()}",
                f"POSTGRES_USER={self.db_user.get()}",
                f"POSTGRES_PASSWORD={self.db_password.get()}",
                f"POSTGRES_HOST_PORT={self.db_host_port.get()}",
                "PIPELINE_MODE=FULL",
                f"PIPELINE_VOXEL_SIZE={self.voxel_size.get()}",
                f"PIPELINE_DB_SRID={self.db_srid.get()}",
                f"PIPELINE_NUM_WORKERS={self.num_workers.get()}",
                f"PIPELINE_OUTPUT_FORMAT={self.output_format.get()}",
                f'RUSTGML2OBJ_EXTRA_FLAGS="{" ".join(gml2obj_flags)}"',
            ]
            if self.processing_mode.get() == "batch":
                lines += [
                    f"BATCH_SOURCE_DIR={self.batch_source_dir.get()}",
                    f"BATCH_MAX_BATCHES={self.batch_max_batches.get()}",
                    f"BATCH_AUTO_ZIP={'true' if self.batch_auto_zip.get() else 'false'}",
                    f"BATCH_OUTPUT_DIR={self.batch_output_dir.get()}",
                ]
            with open(env_path, 'w') as f:
                f.write("\n".join(lines) + "\n")

            self.log_message(f"  Voxel:{self.voxel_size.get()}m  Workers:{self.num_workers.get()}  "
                             f"Output:{self.output_format.get()}  CityGML:{self.citygml_version.get()}", "info")
            self.log_message(f"  GML2OBJ flags: {' '.join(gml2obj_flags) if gml2obj_flags else '(none)'}", "info")
            if self.processing_mode.get() == "batch":
                self.log_message(f"  Batch src: {self.batch_source_dir.get()}", "info")
            self.log_message("✅ .env saved", "success")
            messagebox.showinfo("Success", "Configuration written to .env\n(docker-compose reads it automatically)")
        except Exception as e:
            self.log_message(f"❌ {e}", "error")
            messagebox.showerror("Error", str(e))

    # ========================================================================
    # RUN COMMAND (enhanced — verbose docker, status monitoring)
    # ========================================================================

    def run_command(self, command, description):
        def _run():
            self._command_running = True
            self._last_output_time = time.time()
            self.after(0, lambda: self.start_btn.configure(state="disabled"))

            self.after(0, lambda: self.log_message(f"\n{'═' * 45}", "info"))
            self.after(0, lambda: self.log_message(f"🚀 {description}", "info"))
            self.after(0, lambda: self.log_message(f"{'═' * 45}", "info"))

            try:
                env = os.environ.copy()
                env['COMPOSE_PROGRESS'] = 'plain'
                env['PYTHONUNBUFFERED'] = '1'
                env['DOCKER_CLI_HINTS'] = 'false'

                env_path = os.path.join(self.working_dir, ".env")
                shell_cmd = command
                if os.path.isfile(env_path):
                    shell_cmd = f"set -a && source .env && set +a && {command}"

                process = subprocess.Popen(
                    shell_cmd, shell=True, executable="/bin/bash",
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=self.working_dir, env=env
                )

                for line in process.stdout:
                    self._last_output_time = time.time()
                    stripped = line.rstrip()
                    self.after(0, lambda l=stripped: self._on_pipeline_line(l))

                process.wait()

                if process.returncode == 0:
                    self.after(0, lambda: self.log_message(f"\n✅ {description} completed!", "success"))
                else:
                    self.after(0, lambda: self.log_message(f"\n❌ {description} failed (code {process.returncode})", "error"))

            except Exception as e:
                self.after(0, lambda: self.log_message(f"\n❌ Error: {e}", "error"))
            finally:
                self._command_running = False
                self.after(0, lambda: self.start_btn.configure(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    # Step-detection patterns matched against start.sh stdout
    _STEP_PATTERNS = [
        (r'Step: CityGML 2\.0', 0),         # CityGML 2→3
        (r'Step: CityGML 3\.0', 1),         # OBJ + JSON
        (r'Step: voxelize \+ ingest', 2),   # Voxelize+Ingest
    ]

    def _on_pipeline_line(self, line):
        """Called in main thread for every stdout line — log it and update step indicators."""
        self.log_message(line)
        # Step detection
        for pattern, step_idx in self._STEP_PATTERNS:
            if re.search(pattern, line):
                for i in range(step_idx):
                    self.update_step_status(i, 'complete')
                self.update_step_status(step_idx, 'running')
                return
        if 'Pipeline complete.' in line:
            for i in range(len(self.pipeline_steps)):
                self.update_step_status(i, 'complete')
        elif 'ERROR' in line:
            for i, w in enumerate(self.step_widgets):
                if w['status'].cget('text') == '⏳':
                    self.update_step_status(i, 'error')
                    break

    def _detect_step(self, line):
        """Kept for compatibility — delegates to _on_pipeline_line."""
        self._on_pipeline_line(line)

    # ========================================================================
    # PIPELINE ACTIONS
    # ========================================================================

    def start_pipeline(self):
        if self.processing_mode.get() == "batch":
            src = self.batch_source_dir.get()
            if not src or not os.path.isdir(src):
                messagebox.showerror("Error", "Set a valid source directory for batch processing.")
                return
            self.log_message(f"\n📦 Starting BATCH: {src}", "info")
            self.run_command("./batch-process.sh", "Batch Processing")
        else:
            self.log_message("\n🚀 Starting pipeline...", "info")
            for i in range(len(self.pipeline_steps)):
                self.update_step_status(i, "pending")
            self.run_command("./start.sh", "Voxel Pipeline")

    def clean_data(self):
        """Clean data files using find (handles large directories)"""
        if messagebox.askyesno("Clean Data", "Remove all generated data files?\n(keeps citygml2 input)"):
            self.run_command(
                "find data/citygml3 -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null; "
                "find data/objs -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null; "
                "touch data/citygml3/.gitkeep data/objs/.gitkeep && "
                "echo '✅ Cleaned: data/citygml3/ and data/objs/'",
                "Clean Data Files"
            )

    def clean_docker(self):
        if messagebox.askyesno("Clean Docker", "⚠️ Remove all Docker containers, images, and volumes?"):
            if messagebox.askyesno("Confirm", "This cannot be undone. Are you sure?"):
                self.run_command("bash -c 'echo y | ./clean-all.sh'", "Clean Docker")

    def clear_database(self):
        """Clear database tables — starts PostGIS if not running"""
        if messagebox.askyesno("Clear Database", "Drop all voxel tables?\n(PostGIS will be started if needed)"):
            self.run_command(
                "docker compose up -d postgis && "
                "echo 'Waiting for PostGIS...' && "
                "for i in $(seq 1 20); do "
                "  docker compose exec -T postgis pg_isready -U postgres >/dev/null 2>&1 && break; "
                "  sleep 1; "
                "done && "
                "docker compose exec -T postgis psql -U postgres -d voxel_db "
                "-c 'DROP TABLE IF EXISTS voxel, object_class, object CASCADE;' && "
                "echo '✅ Database tables dropped' || "
                "echo '❌ Failed — check PostGIS status'",
                "Clear Database Tables"
            )

    def restart_database(self):
        """Stop PostGIS, remove its volume, and start fresh"""
        if messagebox.askyesno("Restart Database", "⚠️ This will DELETE all database data and start fresh!"):
            self.run_command(
                "docker compose stop postgis && "
                "docker compose rm -f postgis && "
                "docker volume rm -f voxel_postgis_data 2>/dev/null; "
                "echo 'Starting fresh PostGIS...' && "
                "docker compose up -d postgis && "
                "echo 'Waiting for PostGIS...' && "
                "for i in $(seq 1 20); do "
                "  docker compose exec -T postgis pg_isready -U postgres >/dev/null 2>&1 && break; "
                "  sleep 1; "
                "done && "
                "echo '✅ PostGIS restarted with clean database' || "
                "echo '❌ PostGIS failed to start'",
                "Restart Database"
            )

    # ========================================================================
    # ARCHIVE RESULTS
    # ========================================================================

    def archive_results(self):
        """Archive data/objs → output_batches/<gml_stem>.zip (includes pipeline log)"""
        # Find GML input files to derive the archive name
        objs_dir = os.path.join(self.working_dir, 'data', 'objs')
        citygml2_dir = os.path.join(self.working_dir, 'data', 'citygml2')
        citygml3_dir = os.path.join(self.working_dir, 'data', 'citygml3')

        # Check there are results to archive
        try:
            obj_files = [f for f in os.listdir(objs_dir) if not f.startswith('.')]
            if not obj_files:
                messagebox.showerror("No Results", "data/objs/ is empty — nothing to archive.")
                return
        except Exception:
            messagebox.showerror("Error", "Cannot read data/objs/")
            return

        # Find GML stem for naming
        gml_files = []
        for d in [citygml2_dir, citygml3_dir]:
            try:
                gml_files += [f for f in os.listdir(d) if f.endswith('.gml')]
            except Exception:
                pass

        if not gml_files:
            messagebox.showerror("No Input", "No .gml files found in citygml2/ or citygml3/ to name the archive.")
            return

        gml_stem = os.path.splitext(gml_files[0])[0]
        output_dir = os.path.join(self.working_dir, 'output_batches')
        zip_name = f"{gml_stem}.zip"
        zip_path = os.path.join(output_dir, zip_name)

        if os.path.exists(zip_path):
            if not messagebox.askyesno("Overwrite?", f"{zip_name} already exists in output_batches/.\nOverwrite?"):
                return

        # Save current terminal log into data/objs/ before zipping
        try:
            self.terminal_text.config(state='normal')
            log_content = self.terminal_text.get(1.0, tk.END)
            self.terminal_text.config(state='disabled')
            log_path = os.path.join(objs_dir, 'pipeline_log.txt')
            with open(log_path, 'w') as f:
                f.write(f"AutoVox Pipeline Log — archived {datetime.now().isoformat()}\n")
                f.write(f"GML input: {', '.join(gml_files)}\n")
                f.write("=" * 60 + "\n")
                f.write(log_content)
        except Exception:
            pass  # non-critical

        self.log_message(f"\n📦 Archiving results as {zip_name}...", "info")
        self.log_message(f"   Files: {len(obj_files)} items in data/objs/", "info")

        self.run_command(
            f'mkdir -p output_batches && '
            f'cd data/objs && '
            f'zip -r -q "../../output_batches/{zip_name}" . && '
            f'cd ../.. && '
            f'echo "📦 Archive saved: output_batches/{zip_name}" && '
            f'echo "   Size: $(du -sh "output_batches/{zip_name}" | cut -f1)" && '
            f'echo "   Batch process will skip {gml_stem} on next run"',
            f"Archive → {zip_name}"
        )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        app = ModernVoxelGUI()

        def on_closing():
            try:
                if app.refresh_timer:
                    app.after_cancel(app.refresh_timer)
                app._command_running = False
                app.quit()
                app.destroy()
            except Exception:
                pass

        app.protocol("WM_DELETE_WINDOW", on_closing)
        app.mainloop()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
