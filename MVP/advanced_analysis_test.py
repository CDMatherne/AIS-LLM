#!/usr/bin/env python3
"""
Advanced Analysis Testing Tool for SFD Project
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import traceback
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Advanced_Analysis")

class AdvancedAnalysisTesting:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Analysis Testing Tool")
        self.root.geometry("800x600")
        
        # Set default output directory
        self.output_directory = "C:/AIS_Data/Output"
        
        # Main frame
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        ttk.Label(main_frame, text="Advanced Analysis Testing Tool", 
                 font=("Arial", 16, "bold")).pack(anchor=tk.W)
        ttk.Label(main_frame, text="Access advanced analysis features directly", 
                 font=("Arial", 10)).pack(anchor=tk.W, pady=(0, 15))
        
        # Output directory section
        dir_frame = ttk.Frame(main_frame)
        dir_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(dir_frame, text="Output Directory:").pack(side=tk.LEFT)
        self.output_dir_var = tk.StringVar(value=self.output_directory)
        ttk.Entry(dir_frame, textvariable=self.output_dir_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_frame, text="Browse", command=self.browse_output_dir).pack(side=tk.LEFT)
        
        # Launch button
        ttk.Button(main_frame, text="Launch Advanced Analysis Interface", 
                  command=self.launch_advanced_analysis_direct).pack(pady=20)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_directory = dir_path
            self.output_dir_var.set(dir_path)
    
    def launch_advanced_analysis_direct(self):
        """Direct implementation of advanced analysis interface"""
        try:
            from advanced_analysis import AdvancedAnalysisGUI
            
            # Get output directory
            output_dir = self.output_dir_var.get()
            if not output_dir:
                output_dir = self.output_directory
            
            # Launch the GUI
            advanced_gui = AdvancedAnalysisGUI(self.root, output_dir, 'config.ini')
            self.status_var.set("Advanced Analysis interface launched")
            
        except ImportError as e:
            logger.error(f"Error importing advanced_analysis module: {e}")
            messagebox.showerror("Error", 
                f"Could not import advanced_analysis module:\n{e}\n\n"
                "Please ensure advanced_analysis.py is in the same directory.")
            self._show_fallback_interface()
        except Exception as e:
            logger.error(f"Error launching advanced analysis interface: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Could not launch advanced analysis interface: {e}")
            self._show_fallback_interface()
    
    def _show_fallback_interface(self):
        """Show a fallback interface if the advanced_analysis module can't be loaded"""
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Advanced Analytical Tools")
        analysis_window.geometry("900x700")
        
        # Create main frame with padding
        main_frame = ttk.Frame(analysis_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a header
        ttk.Label(main_frame, text="Advanced Analytical Tools", 
                font=("Arial", 16, "bold")).pack(anchor=tk.W)
        
        # Create a notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create tabs
        tab_output = ttk.Frame(notebook, padding=10)
        tab_analysis = ttk.Frame(notebook, padding=10)
        tab_maps = ttk.Frame(notebook, padding=10)
        tab_vessel = ttk.Frame(notebook, padding=10)
        
        # Add tabs to the notebook
        notebook.add(tab_output, text="Additional Outputs")
        notebook.add(tab_analysis, text="Further Analysis")
        notebook.add(tab_maps, text="Mapping Tools")
        notebook.add(tab_vessel, text="Vessel-Specific Analysis")
        
        # Tab 1: Additional Outputs
        ttk.Label(tab_output, text="Generate Additional Outputs from Dataset", 
                font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        # Add buttons to Tab 1
        output_options = [
            ("Export Full Dataset to CSV", "Export the complete analysis dataset to CSV format"),
            ("Generate Summary Report", "Create a summary report with key findings and statistics"),
            ("Export Vessel Statistics", "Export vessel-specific statistics to Excel format"),
            ("Generate Anomaly Timeline", "Create a timeline visualization of anomalies")
        ]
        
        for btn_text, btn_desc in output_options:
            option_frame = ttk.Frame(tab_output)
            option_frame.pack(fill=tk.X, pady=5)
            
            ttk.Button(option_frame, text=btn_text, width=25,
                    command=lambda t=btn_text: self._show_placeholder(t)).pack(side=tk.LEFT, padx=5)
            
            ttk.Label(option_frame, text=btn_desc, wraplength=500, 
                    font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
    
    def _show_placeholder(self, feature_name):
        """Show a placeholder for features not yet implemented"""
        messagebox.showinfo("Feature Coming Soon", 
                          f"The '{feature_name}' feature will be implemented in the advanced_analysis module.")

def main():
    root = tk.Tk()
    app = AdvancedAnalysisTesting(root)
    root.mainloop()

if __name__ == "__main__":
    main()
