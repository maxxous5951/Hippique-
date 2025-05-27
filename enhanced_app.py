"""
Application complète du prédicteur hippique avec spécialisation Galop/Trot
Version avancée avec modèles séparés pour chaque type de course
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import json
import os
import glob
import pickle
import queue
import threading
import time
import re
import sys
from datetime import datetime, timedelta
import warnings

# Imports ML
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Imports graphiques
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Imports modules locaux AMÉLIORÉS
from cache_manager import IntelligentCache
from error_handler import RobustErrorHandler
from enhanced_feature_engineer import EnhancedFeatureEngineer  # Version améliorée
from enhanced_ensemble_models import SpecializedHorseRacingEnsemble  # Version spécialisée
from backtesting import BacktestingEngine

# Gestion CatBoost optionnel
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost non disponible - utilisation de LightGBM et XGBoost uniquement")

# Gestion psutil optionnel
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil non disponible - monitoring système désactivé")

warnings.filterwarnings('ignore')


class EnhancedHorseRacingGUI:
    """Interface graphique avancée avec spécialisation Galop/Trot"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🏇 Prédicteur Hippique PRO - IA Spécialisée Galop/Trot")
        self.root.geometry("1400x900")
        
        # CORRECTION: Initialiser status_var dès le début
        self.status_var = tk.StringVar()
        self.status_var.set("🚀 Initialisation de l'IA Spécialisée Galop/Trot...")
        
        # Initialisation des composants AMÉLIORÉS
        self.cache = IntelligentCache()
        self.error_handler = RobustErrorHandler()
        self.feature_engineer = EnhancedFeatureEngineer()  # Version améliorée
        self.ensemble = SpecializedHorseRacingEnsemble()   # Version spécialisée
        self.backtesting_engine = BacktestingEngine()
        
        # Données
        self.raw_data = None
        self.processed_data = None
        self.training_results = {'GALOP': {}, 'TROT': {}, 'MIXED': {}}
        self.json_files = []
        self.new_race_data = None
        
        # Statistiques par type de course
        self.race_type_stats = {'GALOP': {}, 'TROT': {}, 'MIXED': {}}
        
        # Queue pour les mises à jour thread-safe
        self.queue = queue.Queue()
        
        # Configuration du style
        self._setup_style()
        
        # Configuration de l'interface
        self.setup_ui()
        self.check_queue()
        
        # Message de bienvenue différé
        self.root.after(1500, self.show_welcome_message)

    def _setup_style(self):
        """Configuration du style visuel moderne"""
        style = ttk.Style()
        style.theme_use('clam')

        # Couleurs modernes avec distinction Galop/Trot
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Galop.TLabel', font=('Arial', 12, 'bold'), foreground='#e74c3c')  # Rouge pour galop
        style.configure('Trot.TLabel', font=('Arial', 12, 'bold'), foreground='#3498db')   # Bleu pour trot
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
        style.configure('Success.TLabel', foreground='#27ae60')
        style.configure('Warning.TLabel', foreground='#e67e22')
        style.configure('Error.TLabel', foreground='#e74c3c')

    def setup_ui(self):
        """Configuration complète de l'interface utilisateur améliorée - ORDRE CORRIGÉ"""
        
        # Header principal avec info spécialisation
        self._create_enhanced_header()
        
        # CORRECTION: Créer la barre de statut EN PREMIER pour initialiser status_var
        self._create_enhanced_status_bar()
        
        # Notebook pour les onglets
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Création des onglets améliorés
        self._create_enhanced_loading_tab()
        self._create_enhanced_training_tab()
        self._create_enhanced_prediction_tab()
        self._create_enhanced_analytics_tab()
        self._create_enhanced_backtesting_tab()
        
        # Démarrage des métriques de performance
        self.update_performance_metrics()

    def _create_enhanced_header(self):
        """Créer l'en-tête amélioré avec spécialisation"""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill='x', padx=10, pady=(10, 0))

        title_label = ttk.Label(header_frame, text="🏇 Prédicteur Hippique PRO - Spécialisé",
                               style='Title.TLabel')
        title_label.pack(side='left')

        # Indicateurs de spécialisation
        spec_frame = ttk.Frame(header_frame)
        spec_frame.pack(side='right')

        ttk.Label(spec_frame, text="🏇 GALOP", style='Galop.TLabel').pack(side='left', padx=10)
        ttk.Label(spec_frame, text="🐎 TROT", style='Trot.TLabel').pack(side='left', padx=5)

        subtitle_label = ttk.Label(header_frame, text="IA Spécialisée : Modèles distincts pour Galop et Trot",
                                  font=('Arial', 10), foreground='#7f8c8d')
        subtitle_label.pack(pady=(5, 0))

    def _create_enhanced_status_bar(self):
        """Créer la barre de statut améliorée"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side='bottom', fill='x', padx=10, pady=5)

        self.status_var = tk.StringVar()
        self.status_var.set("🚀 Prêt - IA Spécialisée Galop/Trot avec modèles séparés")

        status_bar = ttk.Label(status_frame, textvariable=self.status_var,
                              relief='sunken', anchor='w')
        status_bar.pack(side='left', fill='x', expand=True)

        # Indicateurs de performance améliorés
        self.performance_frame = ttk.Frame(status_frame)
        self.performance_frame.pack(side='right')

        # Compteurs par type de course
        self.galop_count_label = ttk.Label(self.performance_frame, text="GALOP: 0",
                                          font=('Arial', 8), foreground='#e74c3c')
        self.galop_count_label.pack(side='right', padx=5)

        self.trot_count_label = ttk.Label(self.performance_frame, text="TROT: 0",
                                         font=('Arial', 8), foreground='#3498db')
        self.trot_count_label.pack(side='right', padx=5)

        self.cache_label = ttk.Label(self.performance_frame, text="Cache: 0%",
                                    font=('Arial', 8), foreground='#7f8c8d')
        self.cache_label.pack(side='right', padx=5)

        self.memory_label = ttk.Label(self.performance_frame, text="RAM: 0MB",
                                     font=('Arial', 8), foreground='#7f8c8d')
        self.memory_label.pack(side='right', padx=5)

    def _create_enhanced_loading_tab(self):
        """Créer l'onglet de chargement amélioré avec stats par type"""
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="📁 Données Spécialisées")

        # Frame principal avec scroll
        canvas = tk.Canvas(self.tab1)
        scrollbar = ttk.Scrollbar(self.tab1, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main_frame = scrollable_frame

        # Titre amélioré
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x', padx=20, pady=20)

        title = ttk.Label(title_frame, text="📁 Chargement Spécialisé des Données",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack()

        subtitle = ttk.Label(title_frame, text="Détection automatique et traitement séparé Galop vs Trot",
                           font=('Arial', 10), foreground='#7f8c8d')
        subtitle.pack(pady=(5, 0))

        # Zone de sélection (identique mais avec nouveau texte)
        self._create_file_selection_section(main_frame)

        # Zone d'informations améliorée
        self._create_enhanced_info_section(main_frame)

        # Barre de progression (identique)
        self._create_progress_section(main_frame)

        # Bouton de traitement amélioré
        self.process_btn = ttk.Button(main_frame, text="🚀 Traiter avec IA Spécialisée Galop/Trot",
                                     command=self.process_files_enhanced,
                                     style='Accent.TButton', state='disabled')
        self.process_btn.pack(pady=20)

        # Statistiques améliorées par type
        self._create_enhanced_stats_section(main_frame)

    def _create_file_selection_section(self, parent):
        """Créer la section de sélection des fichiers"""
        folder_frame = ttk.LabelFrame(parent, text="📂 Dossier des courses (Galop + Trot)", padding=15)
        folder_frame.pack(fill='x', padx=20, pady=10)

        self.folder_var = tk.StringVar()

        folder_input_frame = ttk.Frame(folder_frame)
        folder_input_frame.pack(fill='x')

        folder_entry = ttk.Entry(folder_input_frame, textvariable=self.folder_var,
                               font=('Arial', 10), width=60)
        folder_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))

        browse_btn = ttk.Button(folder_input_frame, text="📂 Parcourir",
                               command=self.browse_folder, style='Accent.TButton')
        browse_btn.pack(side='right')

        scan_btn = ttk.Button(folder_frame, text="🔍 Scanner et Analyser",
                             command=self.scan_folder_enhanced, style='Accent.TButton')
        scan_btn.pack(pady=(10, 0))

    def _create_enhanced_info_section(self, parent):
        """Créer la section d'informations améliorée"""
        info_frame = ttk.LabelFrame(parent, text="ℹ️ Analyse détaillée par type de course", padding=15)
        info_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.info_text = scrolledtext.ScrolledText(info_frame, height=12, width=80,
                                                  font=('Consolas', 9))
        self.info_text.pack(fill='both', expand=True)

    def _create_progress_section(self, parent):
        """Créer la section de progression"""
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill='x', padx=20, pady=10)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100, length=400)
        self.progress_bar.pack(fill='x')

        self.progress_label = ttk.Label(progress_frame, text="", font=('Arial', 9))
        self.progress_label.pack(pady=(5, 0))

    def _create_enhanced_stats_section(self, parent):
        """Créer la section de statistiques améliorée"""
        stats_frame = ttk.LabelFrame(parent, text="📊 Statistiques par type de course", padding=15)
        stats_frame.pack(fill='x', padx=20, pady=10)

        # Statistiques GALOP
        galop_frame = ttk.LabelFrame(stats_frame, text="🏇 GALOP", padding=10)
        galop_frame.pack(fill='x', padx=5, pady=5)

        galop_grid = ttk.Frame(galop_frame)
        galop_grid.pack()

        self.galop_stats = {
            'horses': tk.StringVar(value="0"),
            'races': tk.StringVar(value="0"),
            'avg': tk.StringVar(value="0.0"),
            'features': tk.StringVar(value="0")
        }

        galop_metrics = [
            ("🏇 Chevaux", self.galop_stats['horses'], 0, 0),
            ("🏁 Courses", self.galop_stats['races'], 0, 1),
            ("👥 Moy/course", self.galop_stats['avg'], 0, 2),
            ("🔧 Features", self.galop_stats['features'], 0, 3)
        ]

        for title, var, row, col in galop_metrics:
            frame = ttk.Frame(galop_grid)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky='w')
            ttk.Label(frame, text=title, font=('Arial', 9, 'bold'), foreground='#e74c3c').pack()
            ttk.Label(frame, textvariable=var, font=('Arial', 12, 'bold'), foreground='#c0392b').pack()

        # Statistiques TROT
        trot_frame = ttk.LabelFrame(stats_frame, text="🐎 TROT", padding=10)
        trot_frame.pack(fill='x', padx=5, pady=5)

        trot_grid = ttk.Frame(trot_frame)
        trot_grid.pack()

        self.trot_stats = {
            'horses': tk.StringVar(value="0"),
            'races': tk.StringVar(value="0"),
            'avg': tk.StringVar(value="0.0"),
            'features': tk.StringVar(value="0")
        }

        trot_metrics = [
            ("🐎 Chevaux", self.trot_stats['horses'], 0, 0),
            ("🏁 Courses", self.trot_stats['races'], 0, 1),
            ("👥 Moy/course", self.trot_stats['avg'], 0, 2),
            ("🔧 Features", self.trot_stats['features'], 0, 3)
        ]

        for title, var, row, col in trot_metrics:
            frame = ttk.Frame(trot_grid)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky='w')
            ttk.Label(frame, text=title, font=('Arial', 9, 'bold'), foreground='#3498db').pack()
            ttk.Label(frame, textvariable=var, font=('Arial', 12, 'bold'), foreground='#2980b9').pack()

    def _create_enhanced_training_tab(self):
        """Créer l'onglet d'entraînement amélioré"""
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="🤖 IA Spécialisée")

        # Frame principal avec scroll
        canvas = tk.Canvas(self.tab2)
        scrollbar = ttk.Scrollbar(self.tab2, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main_frame = scrollable_frame

        # Header amélioré
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x', padx=20, pady=20)

        title = ttk.Label(title_frame, text="🤖 Intelligence Artificielle Spécialisée",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack()

        subtitle = ttk.Label(title_frame,
                           text="Modèles séparés et optimisés pour Galop et Trot",
                           font=('Arial', 10), foreground='#7f8c8d')
        subtitle.pack(pady=(5, 0))

        # Information sur la spécialisation
        self._create_specialization_info(main_frame)

        # Configuration d'entraînement (identique)
        self._create_training_config_section(main_frame)

        # Bouton d'entraînement amélioré
        self.train_btn = ttk.Button(main_frame, text="🚀 Entraîner IA Spécialisée (Galop + Trot)",
                                   command=self.train_specialized_models,
                                   style='Accent.TButton', state='disabled')
        self.train_btn.pack(pady=20)

        # Zone de résultats améliorée
        self._create_enhanced_results_section(main_frame)

        # Boutons de gestion améliorés
        self._create_enhanced_model_buttons(main_frame)

    def _create_specialization_info(self, parent):
        """Créer la section d'information sur la spécialisation"""
        info_frame = ttk.LabelFrame(parent, text="🎯 Spécialisations par discipline", padding=15)
        info_frame.pack(fill='x', padx=20, pady=10)

        info_text = """🏇 GALOP - Features spécialisées:
• Poids et handicaps (impact physique direct)
• Œillères et équipement (concentration)
• Distance à l'arrivée (qualité du finish)
• Statut jument pleine (condition physique)

🐎 TROT - Features spécialisées:
• Handicap distance (placement au départ)
• Temps et vitesse (performance chronométrée)
• Réduction kilométrique (performance relative)
• Ferrage (technique de course)
• Avis entraîneur (expertise spécialisée)"""

        ttk.Label(info_frame, text=info_text, justify='left', font=('Arial', 9)).pack(anchor='w')

    def _create_training_config_section(self, parent):
        """Créer la section de configuration d'entraînement"""
        config_frame = ttk.LabelFrame(parent, text="⚙️ Configuration d'entraînement spécialisé", padding=15)
        config_frame.pack(fill='x', padx=20, pady=10)

        config_grid = ttk.Frame(config_frame)
        config_grid.pack(fill='x')

        # Paramètres
        ttk.Label(config_grid, text="Min courses par cheval:").grid(row=0, column=0, sticky='w', padx=5)
        self.min_races_var = tk.IntVar(value=3)
        min_races_spin = ttk.Spinbox(config_grid, from_=1, to=10, textvariable=self.min_races_var, width=10)
        min_races_spin.grid(row=0, column=1, padx=5, sticky='w')

        ttk.Label(config_grid, text="Validations croisées:").grid(row=0, column=2, sticky='w', padx=5)
        self.cv_folds_var = tk.IntVar(value=5)
        cv_folds_spin = ttk.Spinbox(config_grid, from_=3, to=10, textvariable=self.cv_folds_var, width=10)
        cv_folds_spin.grid(row=0, column=3, padx=5, sticky='w')

        # Option d'entraînement séparé
        ttk.Label(config_grid, text="Mode d'entraînement:").grid(row=1, column=0, sticky='w', padx=5)
        self.training_mode_var = tk.StringVar(value="separate")
        training_mode_combo = ttk.Combobox(config_grid, textvariable=self.training_mode_var,
                                          values=['separate', 'mixed'], state='readonly', width=15)
        training_mode_combo.grid(row=1, column=1, padx=5, sticky='w')

        ttk.Label(config_grid, text="separate = Modèles distincts | mixed = Modèle unifié",
                 font=('Arial', 8), foreground='#7f8c8d').grid(row=1, column=2, columnspan=2, sticky='w', padx=5)

    def _create_enhanced_results_section(self, parent):
        """Créer la section de résultats améliorée"""
        results_frame = ttk.LabelFrame(parent, text="📊 Résultats d'entraînement par discipline", padding=15)
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Notebook pour séparer les résultats
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill='both', expand=True)

        # Onglet résultats GALOP
        self.galop_results_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.galop_results_frame, text="🏇 GALOP")

        self.galop_results_text = scrolledtext.ScrolledText(self.galop_results_frame, height=10, font=('Consolas', 9))
        self.galop_results_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Onglet résultats TROT
        self.trot_results_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.trot_results_frame, text="🐎 TROT")

        self.trot_results_text = scrolledtext.ScrolledText(self.trot_results_frame, height=10, font=('Consolas', 9))
        self.trot_results_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Onglet résultats globaux
        self.global_results_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.global_results_frame, text="📊 Comparaison")

        self.global_results_text = scrolledtext.ScrolledText(self.global_results_frame, height=10, font=('Consolas', 9))
        self.global_results_text.pack(fill='both', expand=True, padx=5, pady=5)

    def _create_enhanced_model_buttons(self, parent):
        """Créer les boutons de gestion améliorés"""
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill='x', padx=20, pady=10)

        self.save_btn = ttk.Button(buttons_frame, text="💾 Sauvegarder IA Spécialisée",
                                  command=self.save_specialized_models, state='disabled')
        self.save_btn.pack(side='left', padx=5)

        self.load_btn = ttk.Button(buttons_frame, text="📂 Charger IA Spécialisée",
                                  command=self.load_specialized_models)
        self.load_btn.pack(side='left', padx=5)

        ttk.Label(buttons_frame, text="Format: Modèles spécialisés Galop/Trot avec métriques séparées",
                 font=('Arial', 8), foreground='#7f8c8d').pack(side='left', padx=20)

    def _create_enhanced_prediction_tab(self):
        """Créer l'onglet de prédictions amélioré"""
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="🎯 Prédictions Spécialisées")

        main_frame = ttk.Frame(self.tab3)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Titre amélioré
        title = ttk.Label(main_frame, text="🎯 Prédictions IA Spécialisées",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack(pady=(0, 20))

        # Détection automatique du type
        detection_frame = ttk.LabelFrame(main_frame, text="🔍 Détection automatique du type de course", padding=10)
        detection_frame.pack(fill='x', pady=5)

        self.detected_type_var = tk.StringVar(value="Non détecté")
        ttk.Label(detection_frame, text="Type détecté:").pack(side='left', padx=5)
        self.detected_type_label = ttk.Label(detection_frame, textvariable=self.detected_type_var,
                                           font=('Arial', 10, 'bold'), foreground='#2980b9')
        self.detected_type_label.pack(side='left', padx=10)

        # Mode de prédiction (légèrement modifié)
        self._create_enhanced_prediction_mode_section(main_frame)

        # Bouton de prédiction amélioré
        self.predict_btn = ttk.Button(main_frame, text="🎯 Prédire avec IA Spécialisée",
                                     command=self.predict_race_specialized, style='Accent.TButton')
        self.predict_btn.pack(pady=10)

        # Résultats améliorés (identique mais avec nouveau titre)
        self._create_enhanced_prediction_results_section(main_frame)

    def _create_enhanced_prediction_mode_section(self, parent):
        """Créer la section de mode de prédiction améliorée"""
        mode_frame = ttk.LabelFrame(parent, text="Mode de prédiction", padding=10)
        mode_frame.pack(fill='x', pady=10)

        self.prediction_mode = tk.StringVar(value="existing")

        existing_radio = ttk.Radiobutton(mode_frame, text="📂 Course de la base de données",
                                        variable=self.prediction_mode, value="existing",
                                        command=self.on_prediction_mode_change)
        existing_radio.pack(anchor='w', pady=2)

        new_radio = ttk.Radiobutton(mode_frame, text="📄 Nouveau fichier de course",
                                   variable=self.prediction_mode, value="new",
                                   command=self.on_prediction_mode_change)
        new_radio.pack(anchor='w', pady=2)

        # Frame pour course existante
        self.existing_frame = ttk.LabelFrame(parent, text="Sélection de course existante", padding=10)

        course_select_frame = ttk.Frame(self.existing_frame)
        course_select_frame.pack(fill='x')

        ttk.Label(course_select_frame, text="Course:").pack(side='left', padx=5)
        self.race_var = tk.StringVar()
        self.race_combo = ttk.Combobox(course_select_frame, textvariable=self.race_var,
                                      state='readonly', width=45)
        self.race_combo.pack(side='left', fill='x', expand=True, padx=5)
        self.race_combo.bind('<<ComboboxSelected>>', self.on_race_selection_change)

        # Frame pour nouveau fichier
        self.new_file_frame = ttk.LabelFrame(parent, text="Charger un nouveau fichier", padding=10)

        file_select_frame = ttk.Frame(self.new_file_frame)
        file_select_frame.pack(fill='x', pady=5)

        self.new_file_var = tk.StringVar()
        new_file_entry = ttk.Entry(file_select_frame, textvariable=self.new_file_var, width=50)
        new_file_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))

        browse_new_btn = ttk.Button(file_select_frame, text="📂 Parcourir",
                                   command=self.browse_new_race_file)
        browse_new_btn.pack(side='right')

        load_new_btn = ttk.Button(self.new_file_frame, text="📥 Charger et Analyser",
                                 command=self.load_new_race_file_enhanced)
        load_new_btn.pack(pady=5)

        self.new_file_info = ttk.Label(self.new_file_frame, text="", foreground='green')
        self.new_file_info.pack(pady=5)

        # Initialiser l'affichage
        self.on_prediction_mode_change()

    def _create_enhanced_prediction_results_section(self, parent):
        """Créer la section de résultats de prédiction améliorée"""
        results_frame = ttk.LabelFrame(parent, text="Résultats IA Spécialisée", padding=10)
        results_frame.pack(fill='both', expand=True, pady=10)

        # Tableau des résultats (identique mais avec colonnes améliorées)
        columns = ('Rang', 'N°', 'Nom', 'Driver', 'Prob Victoire', 'Prob Place', 'Cotes', 'Spécialisation', 'Confiance')
        self.prediction_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=12)

        # Variables pour le tri
        self.sort_reverse = {}
        for col in columns:
            self.sort_reverse[col] = False

        # Configuration des colonnes avec tri
        for col in columns:
            self.prediction_tree.heading(col, text=col,
                                       command=lambda c=col: self.sort_treeview_column(c))
            if col in ['Prob Victoire', 'Prob Place', 'Confiance']:
                self.prediction_tree.column(col, width=100)
            elif col == 'Spécialisation':
                self.prediction_tree.column(col, width=90)
            else:
                self.prediction_tree.column(col, width=80)

        scrollbar_pred = ttk.Scrollbar(results_frame, orient='vertical', command=self.prediction_tree.yview)
        self.prediction_tree.configure(yscrollcommand=scrollbar_pred.set)

        self.prediction_tree.pack(side='left', fill='both', expand=True)
        scrollbar_pred.pack(side='right', fill='y')

        # Recommandations IA améliorées
        reco_frame = ttk.LabelFrame(parent, text="Recommandations IA Spécialisées", padding=10)
        reco_frame.pack(fill='x', pady=10)

        self.reco_text = scrolledtext.ScrolledText(reco_frame, height=8, font=('Consolas', 9))
        self.reco_text.pack(fill='both', expand=True)

    def _create_enhanced_analytics_tab(self):
        """Créer l'onglet d'analyses amélioré"""
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="📊 Analytics Spécialisés")

        main_frame = ttk.Frame(self.tab4)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        title = ttk.Label(main_frame, text="📊 Analytics & Comparaison Galop/Trot",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack(pady=(0, 20))

        # Boutons d'analyse améliorés
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill='x', pady=10)

        ttk.Button(buttons_frame, text="📈 Performance par Type",
                  command=self.show_specialized_performance).pack(side='left', padx=5)

        ttk.Button(buttons_frame, text="🔗 Features Spécialisées",
                  command=self.show_specialized_features).pack(side='left', padx=5)

        ttk.Button(buttons_frame, text="⚖️ Comparaison Galop/Trot",
                  command=self.show_galop_trot_comparison).pack(side='left', padx=5)

        ttk.Button(buttons_frame, text="📅 Évolution par Type",
                  command=self.show_temporal_by_type).pack(side='left', padx=5)

        # Zone d'analyse
        self.analysis_frame = ttk.Frame(main_frame)
        self.analysis_frame.pack(fill='both', expand=True, pady=10)

    def _create_enhanced_backtesting_tab(self):
        """Créer l'onglet de backtesting amélioré"""
        self.tab5 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab5, text="📈 Backtesting Spécialisé")

        main_frame = ttk.Frame(self.tab5)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        title = ttk.Label(main_frame, text="📈 Backtesting Spécialisé par Type",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack(pady=(0, 20))

        # Configuration du backtest améliorée
        config_frame = ttk.LabelFrame(main_frame, text="Configuration du backtest spécialisé", padding=15)
        config_frame.pack(fill='x', pady=10)

        config_grid = ttk.Frame(config_frame)
        config_grid.pack()

        ttk.Label(config_grid, text="Période de test (%):").grid(row=0, column=0, sticky='w', padx=5)
        self.backtest_period_var = tk.IntVar(value=20)
        backtest_period_spin = ttk.Spinbox(config_grid, from_=10, to=50,
                                          textvariable=self.backtest_period_var, width=10)
        backtest_period_spin.grid(row=0, column=1, padx=5)

        ttk.Label(config_grid, text="Type de course:").grid(row=0, column=2, sticky='w', padx=5)
        self.backtest_type_var = tk.StringVar(value="all")
        backtest_type_combo = ttk.Combobox(config_grid, textvariable=self.backtest_type_var,
                                          values=['all', 'GALOP', 'TROT'], state='readonly', width=10)
        backtest_type_combo.grid(row=0, column=3, padx=5)

        ttk.Label(config_grid, text="Stratégie:").grid(row=1, column=0, sticky='w', padx=5)
        self.strategy_var = tk.StringVar(value="place_strategy")
        strategy_combo = ttk.Combobox(config_grid, textvariable=self.strategy_var,
                                     values=['place_strategy', 'confidence', 'top_pick', 'value_betting'],
                                     state='readonly', width=15)
        strategy_combo.grid(row=1, column=1, padx=5)

        # Bouton de backtest amélioré
        backtest_btn = ttk.Button(main_frame, text="🚀 Lancer le Backtest Spécialisé",
                                 command=self.run_specialized_backtest, style='Accent.TButton')
        backtest_btn.pack(pady=10)

        # Résultats du backtest avec onglets
        backtest_results_frame = ttk.LabelFrame(main_frame, text="Résultats du backtest", padding=10)
        backtest_results_frame.pack(fill='both', expand=True, pady=10)

        # Notebook pour les résultats
        self.backtest_notebook = ttk.Notebook(backtest_results_frame)
        self.backtest_notebook.pack(fill='both', expand=True)

        # Onglet résultats globaux
        self.backtest_global_frame = ttk.Frame(self.backtest_notebook)
        self.backtest_notebook.add(self.backtest_global_frame, text="📊 Résultats")

        self.backtest_text = scrolledtext.ScrolledText(self.backtest_global_frame, height=12,
                                                      font=('Consolas', 9))
        self.backtest_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Onglet comparaison types
        self.backtest_comparison_frame = ttk.Frame(self.backtest_notebook)
        self.backtest_notebook.add(self.backtest_comparison_frame, text="⚖️ Comparaison")

        self.backtest_comparison_text = scrolledtext.ScrolledText(self.backtest_comparison_frame, height=12,
                                                                 font=('Consolas', 9))
        self.backtest_comparison_text.pack(fill='both', expand=True, padx=5, pady=5)

    # ===== MÉTHODES AMÉLIORÉES =====

    def update_performance_metrics(self):
        """Mettre à jour les métriques de performance améliorées"""
        try:
            # Cache hit rate
            cache_stats = self.cache.get_stats()
            self.cache_label.config(text=f"Cache: {cache_stats['hit_rate']:.0f}%")

            # Utilisation mémoire si psutil disponible
            if PSUTIL_AVAILABLE:
                memory_mb = psutil.Process().memory_info().rss / 1024**2
                self.memory_label.config(text=f"RAM: {memory_mb:.0f}MB")
            else:
                self.memory_label.config(text="RAM: N/A")

            # Compteurs par type de course
            if self.processed_data is not None and 'allure' in self.processed_data.columns:
                galop_count = len(self.processed_data[self.processed_data['allure'] == 'GALOP'])
                trot_count = len(self.processed_data[self.processed_data['allure'] == 'TROT'])
                
                self.galop_count_label.config(text=f"GALOP: {galop_count}")
                self.trot_count_label.config(text=f"TROT: {trot_count}")

        except:
            pass

        # Programmer la prochaine mise à jour
        self.root.after(5000, self.update_performance_metrics)

    def browse_folder(self):
        """Parcourir pour sélectionner un dossier"""
        folder = filedialog.askdirectory(title="Sélectionner le dossier des courses (Galop + Trot)")
        if folder:
            self.folder_var.set(folder)

    def scan_folder_enhanced(self):
        """Scanner le dossier avec analyse spécialisée"""
        folder = self.folder_var.get()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier valide")
            return

        json_files = glob.glob(os.path.join(folder, "*.json"))

        if not json_files:
            messagebox.showwarning("Attention", "Aucun fichier JSON trouvé dans ce dossier")
            return

        self.json_files = json_files

        # Analyse spécialisée par type de course
        info_text = f"✅ {len(json_files)} fichiers JSON trouvés\n\n"
        info_text += "🔍 ANALYSE SPÉCIALISÉE PAR TYPE:\n"
        info_text += "=" * 50 + "\n"

        galop_files = 0
        trot_files = 0
        mixed_files = 0
        total_galop_horses = 0
        total_trot_horses = 0

        sample_size = min(10, len(json_files))

        for i, file in enumerate(json_files[:sample_size]):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                participants = data.get('participants', [])
                filename = os.path.basename(file)
                
                # Analyser le type de course
                allures = set()
                for participant in participants:
                    if 'allure' in participant:
                        allures.add(participant['allure'])
                
                if len(allures) == 1:
                    main_allure = list(allures)[0]
                    if main_allure == 'GALOP':
                        galop_files += 1
                        total_galop_horses += len(participants)
                        info_text += f"🏇 {filename}: {len(participants)} chevaux GALOP\n"
                    elif main_allure == 'TROT':
                        trot_files += 1
                        total_trot_horses += len(participants)
                        info_text += f"🐎 {filename}: {len(participants)} chevaux TROT\n"
                    else:
                        mixed_files += 1
                        info_text += f"❓ {filename}: {len(participants)} chevaux ({main_allure})\n"
                else:
                    mixed_files += 1
                    info_text += f"🔀 {filename}: {len(participants)} chevaux (MIXTE: {', '.join(allures)})\n"
                    
            except Exception as e:
                info_text += f"❌ {os.path.basename(file)}: Erreur de lecture\n"

        if sample_size < len(json_files):
            info_text += f"... et {len(json_files) - sample_size} autres fichiers\n"

        # Estimation globale
        estimated_galop = (galop_files / max(sample_size, 1)) * len(json_files) * (total_galop_horses / max(galop_files, 1) if galop_files > 0 else 0)
        estimated_trot = (trot_files / max(sample_size, 1)) * len(json_files) * (total_trot_horses / max(trot_files, 1) if trot_files > 0 else 0)

        info_text += f"\n📊 ESTIMATION GLOBALE:\n"
        info_text += f"🏇 GALOP: ~{estimated_galop:.0f} chevaux ({galop_files}/{sample_size} fichiers)\n"
        info_text += f"🐎 TROT: ~{estimated_trot:.0f} chevaux ({trot_files}/{sample_size} fichiers)\n"
        info_text += f"🔀 MIXTE: {mixed_files}/{sample_size} fichiers\n"
        info_text += f"\n🔧 Features spécialisées:\n"
        info_text += f"• GALOP: ~50 features (poids, œillères, finish)\n"
        info_text += f"• TROT: ~55 features (temps, ferrage, avis entraîneur)\n"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)

        self.process_btn.config(state='normal')
        self.status_var.set(f"Prêt à traiter {len(json_files)} fichiers avec IA spécialisée Galop/Trot")

    def process_files_enhanced(self):
        """Traiter les fichiers avec spécialisation améliorée"""
        self.process_btn.config(state='disabled')
        self.progress_var.set(0)

        # Lancer le traitement dans un thread séparé
        thread = threading.Thread(target=self._process_files_enhanced_thread)
        thread.daemon = True
        thread.start()

    def _process_files_enhanced_thread(self):
        """Thread de traitement amélioré avec spécialisation"""
        try:
            def _process_internal():
                all_races = []
                total_files = len(self.json_files)

                self.queue.put(('progress', 5, f"Initialisation du traitement IA spécialisé..."))

                for i, file_path in enumerate(self.json_files):
                    progress = (i + 1) / total_files * 70  # 70% pour le chargement
                    filename = os.path.basename(file_path)

                    self.queue.put(('progress', progress, f"Traitement spécialisé: {filename} ({i+1}/{total_files})"))

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            race_data = json.load(f)

                        # Extraction de la date du nom de fichier
                        date_match = re.search(r'(\d{8})', filename)
                        if date_match:
                            race_date = datetime.strptime(date_match.group(1), '%d%m%Y')
                        else:
                            race_date = datetime.now()

                        # Traitement des participants
                        participants = race_data.get('participants', [])
                        for participant in participants:
                            participant['race_date'] = race_date
                            participant['race_file'] = filename
                            all_races.append(participant)

                    except Exception as e:
                        self.queue.put(('warning', f"Erreur avec {filename}: {str(e)}"))
                        continue

                if not all_races:
                    raise ValueError("Aucune donnée n'a pu être chargée")

                # Création du DataFrame
                self.queue.put(('progress', 75, "Création du DataFrame..."))
                self.raw_data = pd.DataFrame(all_races)

                # Extraction des features SPÉCIALISÉES
                self.queue.put(('progress', 85, "Extraction des features IA spécialisées Galop/Trot..."))
                self.processed_data = self.feature_engineer.extract_comprehensive_features_by_type(self.raw_data)

                # Optimisation mémoire
                self.queue.put(('progress', 95, "Optimisation mémoire..."))
                self.processed_data = self.optimize_dataframe_memory(self.processed_data)

                # Statistiques finales PAR TYPE
                self._calculate_specialized_stats()

                self.queue.put(('progress', 100, "Traitement IA spécialisé terminé !"))
                self.queue.put(('complete', "Données traitées avec succès par l'IA spécialisée !"))

            # Exécution avec gestion d'erreurs robuste
            self.error_handler.robust_execute(_process_internal, 'data_loading')

        except Exception as e:
            self.queue.put(('error', f"Erreur lors du traitement IA spécialisé: {str(e)}"))

    def _calculate_specialized_stats(self):
        """Calculer les statistiques spécialisées par type"""
        if self.processed_data is None or 'allure' not in self.processed_data.columns:
            return

        # Stats globales
        total_horses = len(self.processed_data)
        unique_races = self.processed_data['race_file'].nunique()
        date_range = (self.processed_data['race_date'].max() - self.processed_data['race_date'].min()).days
        
        # Stats par type
        for race_type in ['GALOP', 'TROT']:
            type_data = self.processed_data[self.processed_data['allure'] == race_type]
            
            if len(type_data) > 0:
                type_horses = len(type_data)
                type_races = type_data['race_file'].nunique()
                type_avg = type_horses / max(type_races, 1)
                type_features = type_data.shape[1]
                
                self.race_type_stats[race_type] = {
                    'horses': type_horses,
                    'races': type_races,
                    'avg': type_avg,
                    'features': type_features
                }
                
                # Mettre à jour l'interface
                self.queue.put(('specialized_stats', race_type, type_horses, type_races, type_avg, type_features))

    def optimize_dataframe_memory(self, df):
        """Optimiser l'usage mémoire du DataFrame (identique)"""
        original_memory = df.memory_usage(deep=True).sum()

        # Optimiser les types numériques
        for col in df.select_dtypes(include=['int64']).columns:
            col_min, col_max = df[col].min(), df[col].max()

            if col_min >= 0:
                if col_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)

        # Optimiser les floats
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        optimized_memory = df.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100

        if PSUTIL_AVAILABLE:
            try:
                current_memory = psutil.Process().memory_info().rss / 1024**2
                print(f"💾 Optimisation mémoire: {reduction:.1f}% de réduction - RAM actuelle: {current_memory:.0f}MB")
            except:
                print(f"💾 Optimisation mémoire: {reduction:.1f}% de réduction")

        return df

    def train_specialized_models(self):
        """Entraîner les modèles spécialisés"""
        if self.processed_data is None:
            messagebox.showerror("Erreur", "Veuillez d'abord charger les données")
            return

        self.train_btn.config(state='disabled')
        
        # Vider les zones de résultats
        self.galop_results_text.delete(1.0, tk.END)
        self.trot_results_text.delete(1.0, tk.END)
        self.global_results_text.delete(1.0, tk.END)

        # Lancer l'entraînement dans un thread
        thread = threading.Thread(target=self._train_specialized_thread)
        thread.daemon = True
        thread.start()

    def _train_specialized_thread(self):
        """Thread d'entraînement spécialisé"""
        try:
            def _train_internal():
                min_races = self.min_races_var.get()
                cv_folds = self.cv_folds_var.get()
                training_mode = self.training_mode_var.get()

                # Filtrage des données de base
                filtered_data = self.processed_data[self.processed_data['nombreCourses'] >= min_races].copy()

                self.queue.put(('training_info', 'global',
                    f"🤖 ENTRAÎNEMENT IA SPÉCIALISÉ GALOP/TROT\n"
                    f"{'='*70}\n"
                    f"Dataset: {len(filtered_data)} chevaux (min {min_races} courses)\n"
                    f"Mode: {training_mode}\n"
                    f"Validation: {cv_folds} folds temporels\n\n"
                ))

                if training_mode == 'separate':
                    # Entraînement séparé par type
                    all_results = self.ensemble.train_all_race_types(filtered_data, self.feature_engineer, cv_folds)
                    
                    # Afficher les résultats par type
                    for race_type, results in all_results.items():
                        self.queue.put(('training_info', race_type.lower(),
                            f"🎯 RÉSULTATS {race_type}\n"
                            f"{'='*40}\n"
                        ))
                        
                        for target_name, cv_scores in results.items():
                            self.queue.put(('training_info', race_type.lower(),
                                f"\n📊 {target_name.upper()}:\n"
                            ))
                            
                            for model_name, score in cv_scores.items():
                                self.queue.put(('training_info', race_type.lower(),
                                    f"  • {model_name}: {score:.4f}\n"
                                ))
                
                else:
                    # Entraînement unifié (mode mixte)
                    messagebox.showinfo("Info", "Mode unifié pas encore implémenté - utilisation du mode séparé")
                    return

                # Marquer comme entraîné
                self.training_results = all_results
                
                # Résumé global
                summary = self.ensemble.get_training_summary()
                self.queue.put(('training_info', 'global',
                    f"\n🏆 RÉSUMÉ GLOBAL\n"
                    f"{'='*30}\n"
                    f"Types entraînés: {', '.join(summary['race_types_trained'])}\n"
                    f"Modèles totaux: {summary['total_models']}\n\n"
                ))

                for race_type, info in summary['performance_by_type'].items():
                    self.queue.put(('training_info', 'global',
                        f"📊 {race_type}: {info['model_count']} modèles\n"
                    ))
                    for target, score in info['performance'].items():
                        self.queue.put(('training_info', 'global',
                            f"  • {target}: {score:.4f}\n"
                        ))

                self.queue.put(('specialized_training_complete',
                    "🎉 Entraînement IA Spécialisé terminé avec succès !\n\n"
                    "Les modèles Galop et Trot sont prêts à prédire les courses."
                ))

            # Exécution avec gestion d'erreurs
            self.error_handler.robust_execute(_train_internal, 'model_training')

        except Exception as e:
            self.queue.put(('training_error', f"Erreur d'entraînement IA spécialisé: {str(e)}"))

    # ===== MÉTHODES DE PRÉDICTION SPÉCIALISÉES =====

    def on_prediction_mode_change(self):
        """Gérer le changement de mode de prédiction"""
        mode = self.prediction_mode.get()

        if mode == "existing":
            self.existing_frame.pack(fill='x', pady=10)
            self.new_file_frame.pack_forget()

            # Remplir la liste des courses si disponible
            if self.processed_data is not None:
                race_files = sorted(self.processed_data['race_file'].unique())
                self.race_combo['values'] = race_files
                if race_files:
                    self.race_combo.set(race_files[0])
                    self.on_race_selection_change()
        else:
            self.existing_frame.pack_forget()
            self.new_file_frame.pack(fill='x', pady=10)

    def on_race_selection_change(self, event=None):
        """Détecter le type de course sélectionnée"""
        if self.processed_data is None:
            return
            
        race_file = self.race_var.get()
        if not race_file:
            return

        race_data = self.processed_data[self.processed_data['race_file'] == race_file]
        if not race_data.empty and 'allure' in race_data.columns:
            detected_type = race_data['allure'].iloc[0]
            self.detected_type_var.set(f"{detected_type} 🎯")
            
            # Changer la couleur selon le type
            if detected_type == 'GALOP':
                self.detected_type_label.config(foreground='#e74c3c')
            elif detected_type == 'TROT':
                self.detected_type_label.config(foreground='#3498db')
        else:
            self.detected_type_var.set("Non détecté")

    def browse_new_race_file(self):
        """Parcourir pour sélectionner un nouveau fichier de course"""
        filename = filedialog.askopenfilename(
            title="Sélectionner un fichier de course (Galop ou Trot)",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.new_file_var.set(filename)

    def load_new_race_file_enhanced(self):
        """Charger un nouveau fichier avec détection de type"""
        filename = self.new_file_var.get()

        if not filename or not os.path.exists(filename):
            messagebox.showerror("Erreur", "Veuillez sélectionner un fichier valide")
            return

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                race_data = json.load(f)

            basename = os.path.basename(filename)
            date_match = re.search(r'(\d{8})', basename)
            race_date = datetime.strptime(date_match.group(1), '%d%m%Y') if date_match else datetime.now()

            participants = []
            detected_types = set()
            
            for participant in race_data['participants']:
                participant['race_date'] = race_date
                participant['race_file'] = basename
                participants.append(participant)
                
                # Détecter le type
                if 'allure' in participant:
                    detected_types.add(participant['allure'])

            temp_df = pd.DataFrame(participants)

            # Réinitialiser le détecteur de nouvelles valeurs
            if hasattr(self.feature_engineer, '_unknown_detected'):
                delattr(self.feature_engineer, '_unknown_detected')

            # Extraction des features SPÉCIALISÉES
            print(f"🔄 Traitement spécialisé du fichier: {basename}")
            print(f"📊 Participants trouvés: {len(participants)}")
            print(f"🎯 Types détectés: {', '.join(detected_types)}")

            # Extraire les features avec la méthode spécialisée
            self.new_race_data = self.feature_engineer.extract_comprehensive_features_by_type(temp_df)

            num_horses = len(self.new_race_data)
            num_features = self.new_race_data.shape[1]
            main_type = list(detected_types)[0] if len(detected_types) == 1 else "MIXTE"

            # Mettre à jour la détection
            self.detected_type_var.set(f"{main_type} 🎯")
            if main_type == 'GALOP':
                self.detected_type_label.config(foreground='#e74c3c')
            elif main_type == 'TROT':
                self.detected_type_label.config(foreground='#3498db')

            # Vérifier s'il y a de nouvelles valeurs
            unknown_report = self.feature_engineer.get_unknown_values_report()

            if unknown_report:
                info_message = (
                    f"✅ Fichier {main_type} chargé avec succès !\n\n"
                    f"📊 STATISTIQUES:\n"
                    f"• {num_horses} chevaux traités\n"
                    f"• {num_features} caractéristiques spécialisées\n"
                    f"• Type détecté: {main_type}\n\n"
                    f"⚠️ NOUVELLES VALEURS DÉTECTÉES:\n"
                    f"L'IA a détecté de nouveaux participants non vus\n"
                    f"pendant l'entraînement. Voir la console pour détails."
                )

                print(f"\n{unknown_report}")
                messagebox.showinfo("Succès avec nouvelles valeurs", info_message)
                self.new_file_info.config(text=f"⚠️ {main_type} chargé avec nouveaux participants: {num_horses} chevaux")
            else:
                messagebox.showinfo("Succès",
                    f"✅ Fichier {main_type} chargé parfaitement !\n\n"
                    f"• {num_horses} chevaux traités par l'IA spécialisée\n"
                    f"• {num_features} caractéristiques {main_type} extraites\n"
                    f"• Tous les participants sont connus de l'IA\n"
                    f"• Prêt pour prédiction spécialisée optimale"
                )
                self.new_file_info.config(text=f"✅ {main_type} chargé: {num_horses} chevaux - {basename}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement spécialisé: {str(e)}")
            self.new_file_info.config(text="❌ Erreur de chargement")

    def predict_race_specialized(self):
        """Prédire une course avec l'IA spécialisée"""
        # Vérifier qu'au moins un modèle est entraîné
        if not any(self.ensemble.is_trained.values()):
            messagebox.showerror("Erreur", "Veuillez d'abord entraîner l'IA Spécialisée")
            return

        mode = self.prediction_mode.get()
        race_data = None

        if mode == "existing":
            if self.processed_data is None:
                messagebox.showerror("Erreur", "Aucune donnée de base chargée")
                return

            race_file = self.race_var.get()
            if not race_file:
                messagebox.showwarning("Attention", "Veuillez sélectionner une course")
                return

            race_data = self.processed_data[self.processed_data['race_file'] == race_file].copy()
            if race_data.empty:
                messagebox.showerror("Erreur", "Aucune donnée trouvée pour cette course")
                return
        else:
            if self.new_race_data is None:
                messagebox.showwarning("Attention", "Veuillez d'abord charger un fichier de course")
                return
            race_data = self.new_race_data.copy()

        try:
            # Détecter le type de course
            race_type = self.ensemble.get_best_race_type_for_prediction(race_data)
            print(f"🎯 Type de course détecté: {race_type}")

            # Vérifier que le modèle correspondant est entraîné
            if not self.ensemble.is_trained[race_type]:
                available_types = [rt for rt, trained in self.ensemble.is_trained.items() if trained]
                if not available_types:
                    messagebox.showerror("Erreur", "Aucun modèle entraîné disponible")
                    return
                
                race_type = available_types[0]
                messagebox.showinfo("Info", 
                    f"Type spécifique non entraîné, utilisation du modèle {race_type}")

            # Obtenir les features spécialisées
            feature_names = self.ensemble.feature_names[race_type]
            if not feature_names:
                messagebox.showerror("Erreur", f"Aucune feature définie pour {race_type}")
                return

            # Vérifier les features
            missing_features = [f for f in feature_names if f not in race_data.columns]
            if missing_features:
                messagebox.showerror("Erreur", f"Features {race_type} manquantes: {missing_features[:5]}...")
                return

            # Préparation des données
            X_pred = race_data[feature_names].fillna(0)

            # Prédictions spécialisées
            predictions = {}
            confidence_scores = {}

            for target_name in ['win', 'place', 'position']:
                try:
                    pred = self.ensemble.predict_specialized_ensemble(X_pred, target_name, race_type)
                    predictions[target_name] = pred

                    # Calcul de confiance spécialisée
                    confidence = self.ensemble.calculate_specialized_confidence(X_pred, target_name, race_type)
                    confidence_scores[target_name] = confidence

                except Exception as e:
                    print(f"Erreur prédiction {target_name}: {str(e)}")
                    continue

            if not predictions:
                messagebox.showerror("Erreur", "Aucune prédiction n'a pu être générée")
                return

            # Préparation des résultats améliorés
            results_df = race_data[['numPmu', 'nom', 'driver', 'direct_odds']].copy()

            # Ajouter les probabilités
            if 'position' in predictions:
                results_df['pred_position'] = predictions['position']
                results_df = results_df.sort_values('pred_position')
            else:
                results_df = results_df.sort_values('prob_win', ascending=False)

            results_df['prob_win'] = predictions.get('win', np.zeros(len(results_df)))
            results_df['prob_place'] = predictions.get('place', np.zeros(len(results_df)))

            # Score de confiance global
            all_confidences = []
            for target in confidence_scores.values():
                all_confidences.append(target)

            if all_confidences:
                results_df['confidence'] = np.mean(all_confidences, axis=0)
            else:
                results_df['confidence'] = np.ones(len(results_df)) * 0.5

            # Réinitialiser l'index pour le classement
            results_df = results_df.reset_index(drop=True)

            # Affichage dans le tableau AMÉLIORÉ
            self.prediction_tree.delete(*self.prediction_tree.get_children())

            for i, (_, horse) in enumerate(results_df.iterrows()):
                confidence_pct = horse['confidence'] * 100

                # Code couleur pour la confiance
                if confidence_pct >= 80:
                    confidence_text = f"{confidence_pct:.0f}% 🔥"
                elif confidence_pct >= 60:
                    confidence_text = f"{confidence_pct:.0f}% ✅"
                elif confidence_pct >= 40:
                    confidence_text = f"{confidence_pct:.0f}% ⚠️"
                else:
                    confidence_text = f"{confidence_pct:.0f}% ❓"

                # Icône de spécialisation
                specialization_icon = "🏇" if race_type == 'GALOP' else "🐎" if race_type == 'TROT' else "🔀"

                self.prediction_tree.insert('', 'end', values=(
                    i + 1,
                    horse['numPmu'],
                    horse['nom'][:20],
                    horse['driver'][:15],
                    f"{horse['prob_win']:.3f}",
                    f"{horse['prob_place']:.3f}",
                    f"{horse['direct_odds']:.1f}",
                    f"{race_type} {specialization_icon}",
                    confidence_text
                ))

            # Générer les recommandations IA spécialisées
            self.generate_specialized_ai_recommendations(results_df, race_type)

            # Mise à jour du statut
            source_info = "Course de la base" if mode == "existing" else "Nouveau fichier"
            self.status_var.set(f"Prédiction IA {race_type} réalisée - {source_info}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la prédiction spécialisée: {str(e)}")

    def generate_specialized_ai_recommendations(self, results_df, race_type):
        """Générer les recommandations IA spécialisées"""
        reco_text = f"🤖 RECOMMANDATIONS IA SPÉCIALISÉES {race_type} 🤖\n"
        reco_text += "=" * 70 + "\n\n"

        # Type de course et spécificités
        if race_type == 'GALOP':
            reco_text += f"🏇 ANALYSE GALOP - Facteurs clés:\n"
            reco_text += f"• Poids et handicaps (impact physique)\n"
            reco_text += f"• Qualité du finish (distance à l'arrivée)\n"
            reco_text += f"• Équipement (œillères, condition jument)\n\n"
        elif race_type == 'TROT':
            reco_text += f"🐎 ANALYSE TROT - Facteurs clés:\n"
            reco_text += f"• Performance chronométrée (temps/vitesse)\n"
            reco_text += f"• Technique de course (ferrage, avis entraîneur)\n"
            reco_text += f"• Handicap distance (placement départ)\n\n"

        # Analyse du favori IA spécialisé
        best_pick = results_df.iloc[0]
        reco_text += f"🏆 FAVORI IA {race_type}\n"
        reco_text += f"{'='*25}\n"

        confidence_level = best_pick['confidence']
        if confidence_level >= 0.8:
            confidence_icon = "🔥 TRÈS HAUTE"
        elif confidence_level >= 0.6:
            confidence_icon = "✅ HAUTE"
        elif confidence_level >= 0.4:
            confidence_icon = "⚠️ MOYENNE"
        else:
            confidence_icon = "❓ FAIBLE"

        reco_text += f"#{best_pick['numPmu']} - {best_pick['nom']}\n"
        reco_text += f"Probabilité victoire: {best_pick['prob_win']:.1%}\n"
        reco_text += f"Probabilité place: {best_pick['prob_place']:.1%}\n"
        reco_text += f"Confiance IA {race_type}: {confidence_icon} ({confidence_level:.1%})\n"
        reco_text += f"Cote: {best_pick['direct_odds']:.1f}\n\n"

        # Tiercé IA spécialisé
        reco_text += f"🎯 TIERCÉ {race_type} RECOMMANDÉ\n"
        reco_text += f"{'='*35}\n"
        top3 = results_df.head(3)

        for i, (_, horse) in enumerate(top3.iterrows()):
            conf_pct = horse['confidence'] * 100
            reco_text += f"{i+1}. #{horse['numPmu']} - {horse['nom']} "
            reco_text += f"({conf_pct:.0f}% confiance {race_type})\n"

        # Analyse des outsiders spécialisés
        reco_text += f"\n💎 OUTSIDERS {race_type} À SURVEILLER\n"
        reco_text += f"{'='*40}\n"

        outsiders = results_df[(results_df['direct_odds'] > 10) &
                              (results_df['prob_win'] > 0.05)]

        if not outsiders.empty:
            best_outsider = outsiders.loc[outsiders['prob_win'].idxmax()]
            expected_value = best_outsider['prob_win'] * best_outsider['direct_odds']

            if expected_value > 1.2:
                reco_text += f"#{best_outsider['numPmu']} - {best_outsider['nom']}\n"
                reco_text += f"Cote: {best_outsider['direct_odds']:.1f} | "
                reco_text += f"Prob: {best_outsider['prob_win']:.1%} | "
                reco_text += f"Valeur {race_type}: {expected_value:.2f}\n"
            else:
                reco_text += f"Aucun outsider {race_type} intéressant détecté\n"
        else:
            reco_text += f"Aucun outsider {race_type} dans les critères\n"

        # Stratégie spécialisée
        reco_text += f"\n📈 STRATÉGIE IA {race_type}\n"
        reco_text += f"{'='*30}\n"

        avg_confidence = results_df['confidence'].mean()

        if avg_confidence > 0.7:
            strategy = "🔥 EXCELLENTE"
        elif avg_confidence > 0.6:
            strategy = "✅ BONNE"
        elif avg_confidence > 0.5:
            strategy = "⚠️ CORRECTE"
        else:
            strategy = "❌ À AMÉLIORER"

        reco_text += f"Qualité analyse {race_type}: {strategy}\n"
        reco_text += f"Confiance moyenne: {avg_confidence:.1%}\n"

        # Conseils spécialisés
        if race_type == 'GALOP':
            reco_text += f"\n💡 CONSEILS GALOP:\n"
            reco_text += f"• Surveiller les poids légers sur longue distance\n"
            reco_text += f"• Les œillères peuvent améliorer la concentration\n"
            reco_text += f"• Attention aux juments pleines (baisse de forme)\n"
        elif race_type == 'TROT':
            reco_text += f"\n💡 CONSEILS TROT:\n"
            reco_text += f"• Les temps précédents sont très indicatifs\n"
            reco_text += f"• L'avis entraîneur a un poids important\n"
            reco_text += f"• Le ferrage influence la technique de course\n"

        # Footer technique spécialisé
        reco_text += f"\n🔬 Powered by IA Spécialisée {race_type}:\n"
        
        # Compter les modèles utilisés
        models_count = len(self.ensemble.models.get(race_type, {}).get('win', {}))
        features_count = len(self.ensemble.feature_names.get(race_type, []))
        
        reco_text += f"Modèles {race_type}: {models_count} IA combinées\n"
        reco_text += f"Features {race_type}: {features_count} spécialisées\n"

        self.reco_text.delete(1.0, tk.END)
        self.reco_text.insert(1.0, reco_text)

    def sort_treeview_column(self, col):
        """Trier le tableau par colonne (identique à la version de base)"""
        data = [(self.prediction_tree.set(child, col), child) 
                for child in self.prediction_tree.get_children('')]

        # Inverser l'ordre si déjà trié
        reverse = self.sort_reverse[col]
        
        # Tri numérique pour certaines colonnes
        if col in ['Rang', 'N°', 'Prob Victoire', 'Prob Place', 'Cotes']:
            try:
                data.sort(key=lambda x: float(x[0].replace('%', '').replace('🔥', '').replace('✅', '').replace('⚠️', '').replace('❓', '')), reverse=reverse)
            except:
                data.sort(reverse=reverse)
        else:
            data.sort(reverse=reverse)

        for index, (val, child) in enumerate(data):
            self.prediction_tree.move(child, '', index)

        # Inverser pour le prochain clic
        self.sort_reverse[col] = not reverse

    # ===== MÉTHODES D'ANALYTICS SPÉCIALISÉES =====

    def show_specialized_performance(self):
        """Afficher les performances spécialisées par type"""
        if not self.training_results:
            messagebox.showwarning("Attention", "Aucun modèle spécialisé entraîné")
            return

        # Nettoyer le frame
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()

        # Créer les graphiques spécialisés
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance des Modèles IA Spécialisés Galop/Trot', fontsize=16, fontweight='bold')

        # Graphique 1: Comparaison des performances par type
        race_types = list(self.training_results.keys())
        performance_data = {}
        
        for race_type in race_types:
            if self.training_results[race_type]:
                for target, results in self.training_results[race_type].items():
                    if target not in performance_data:
                        performance_data[target] = {}
                    performance_data[target][race_type] = max(results.values()) if results else 0

        if performance_data:
            targets = list(performance_data.keys())
            x_pos = np.arange(len(targets))
            width = 0.35

            for i, race_type in enumerate(race_types):
                scores = [performance_data[target].get(race_type, 0) for target in targets]
                color = '#e74c3c' if race_type == 'GALOP' else '#3498db' if race_type == 'TROT' else '#95a5a6'
                axes[0, 0].bar(x_pos + i*width, scores, width, label=race_type, color=color, alpha=0.8)

            axes[0, 0].set_xlabel('Objectifs')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Comparaison Performance par Type')
            axes[0, 0].set_xticks(x_pos + width/2)
            axes[0, 0].set_xticklabels(targets)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Graphique 2: Répartition des modèles par type
        model_counts = {}
        for race_type in race_types:
            if race_type in self.ensemble.models:
                count = 0
                for target_models in self.ensemble.models[race_type].values():
                    count += len(target_models)
                model_counts[race_type] = count

        if model_counts:
            colors = ['#e74c3c' if rt == 'GALOP' else '#3498db' if rt == 'TROT' else '#95a5a6' 
                     for rt in model_counts.keys()]
            axes[0, 1].pie(model_counts.values(), labels=model_counts.keys(), 
                          colors=colors, autopct='%1.1f%%')
            axes[0, 1].set_title('Répartition des Modèles')

        # Graphique 3: Performance détaillée GALOP
        if 'GALOP' in self.training_results and self.training_results['GALOP']:
            galop_data = self.training_results['GALOP']
            galop_scores = []
            galop_labels = []
            
            for target, models in galop_data.items():
                for model, score in models.items():
                    galop_scores.append(score)
                    galop_labels.append(f"{target}_{model}")
            
            if galop_scores:
                axes[1, 0].barh(range(len(galop_scores)), galop_scores, color='#e74c3c', alpha=0.7)
                axes[1, 0].set_yticks(range(len(galop_labels)))
                axes[1, 0].set_yticklabels([label[:10] for label in galop_labels])
                axes[1, 0].set_xlabel('Score')
                axes[1, 0].set_title('Détail Performance GALOP')
                axes[1, 0].grid(True, alpha=0.3)

        # Graphique 4: Performance détaillée TROT
        if 'TROT' in self.training_results and self.training_results['TROT']:
            trot_data = self.training_results['TROT']
            trot_scores = []
            trot_labels = []
            
            for target, models in trot_data.items():
                for model, score in models.items():
                    trot_scores.append(score)
                    trot_labels.append(f"{target}_{model}")
            
            if trot_scores:
                axes[1, 1].barh(range(len(trot_scores)), trot_scores, color='#3498db', alpha=0.7)
                axes[1, 1].set_yticks(range(len(trot_labels)))
                axes[1, 1].set_yticklabels([label[:10] for label in trot_labels])
                axes[1, 1].set_xlabel('Score')
                axes[1, 1].set_title('Détail Performance TROT')
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Intégrer dans tkinter
        canvas = FigureCanvasTkAgg(fig, self.analysis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def show_temporal_by_type(self):
        """Afficher l'analyse temporelle par type de course"""
        if self.processed_data is None:
            messagebox.showwarning("Attention", "Aucune donnée chargée")
            return

        # Nettoyer le frame
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Évolution Temporelle par Type de Course', fontsize=16, fontweight='bold')

        try:
            # Séparer les données par type
            galop_data = self.processed_data[self.processed_data['allure'] == 'GALOP']
            trot_data = self.processed_data[self.processed_data['allure'] == 'TROT']

            # Graphique 1: Évolution mensuelle des performances GALOP
            if not galop_data.empty:
                galop_monthly = galop_data.groupby(
                    galop_data['race_date'].dt.to_period('M')
                ).agg({
                    'win_rate': 'mean',
                    'direct_odds': 'median'
                }).reset_index()

                if not galop_monthly.empty:
                    months_galop = [str(period) for period in galop_monthly['race_date']]
                    axes[0, 0].plot(months_galop, galop_monthly['win_rate'], 
                                   marker='o', linewidth=2, color='#e74c3c', markersize=6)
                    axes[0, 0].set_title('Évolution GALOP - Taux Victoire')
                    axes[0, 0].set_ylabel('Taux de Victoire')
                    axes[0, 0].tick_params(axis='x', rotation=45)
                    axes[0, 0].grid(True, alpha=0.3)

            # Graphique 2: Évolution mensuelle des performances TROT
            if not trot_data.empty:
                trot_monthly = trot_data.groupby(
                    trot_data['race_date'].dt.to_period('M')
                ).agg({
                    'win_rate': 'mean',
                    'direct_odds': 'median'
                }).reset_index()

                if not trot_monthly.empty:
                    months_trot = [str(period) for period in trot_monthly['race_date']]
                    axes[0, 1].plot(months_trot, trot_monthly['win_rate'], 
                                   marker='s', linewidth=2, color='#3498db', markersize=6)
                    axes[0, 1].set_title('Évolution TROT - Taux Victoire')
                    axes[0, 1].set_ylabel('Taux de Victoire')
                    axes[0, 1].tick_params(axis='x', rotation=45)
                    axes[0, 1].grid(True, alpha=0.3)

            # Graphique 3: Comparaison saisonnière
            if not galop_data.empty and not trot_data.empty:
                galop_seasonal = galop_data.groupby(galop_data['race_date'].dt.month)['win_rate'].mean()
                trot_seasonal = trot_data.groupby(trot_data['race_date'].dt.month)['win_rate'].mean()

                months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                         'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
                
                x_pos = np.arange(1, 13)
                width = 0.35

                axes[1, 0].bar(x_pos - width/2, [galop_seasonal.get(i, 0) for i in x_pos], 
                              width, label='GALOP', color='#e74c3c', alpha=0.8)
                axes[1, 0].bar(x_pos + width/2, [trot_seasonal.get(i, 0) for i in x_pos], 
                              width, label='TROT', color='#3498db', alpha=0.8)
                
                axes[1, 0].set_xlabel('Mois')
                axes[1, 0].set_ylabel('Taux de Victoire')
                axes[1, 0].set_title('Comparaison Saisonnière')
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels(months, rotation=45)
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Graphique 4: Distribution par jour de la semaine
            if 'day_of_week' in self.processed_data.columns:
                day_names = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
                
                galop_days = galop_data['day_of_week'].value_counts().sort_index()
                trot_days = trot_data['day_of_week'].value_counts().sort_index()

                x_pos = np.arange(len(day_names))
                width = 0.35

                axes[1, 1].bar(x_pos - width/2, [galop_days.get(i, 0) for i in range(7)], 
                              width, label='GALOP', color='#e74c3c', alpha=0.8)
                axes[1, 1].bar(x_pos + width/2, [trot_days.get(i, 0) for i in range(7)], 
                              width, label='TROT', color='#3498db', alpha=0.8)
                
                axes[1, 1].set_xlabel('Jour de la Semaine')
                axes[1, 1].set_ylabel('Nombre de Courses')
                axes[1, 1].set_title('Répartition Hebdomadaire')
                axes[1, 1].set_xticks(x_pos)
                axes[1, 1].set_xticklabels(day_names)
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

        except Exception as e:
            for ax in axes.flat:
                ax.text(0.5, 0.5, f'Erreur:\n{str(e)}',
                       ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.analysis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    # ===== MÉTHODES DE BACKTESTING SPÉCIALISÉES =====

    def run_specialized_backtest(self):
        """Lancer le backtesting spécialisé"""
        if not any(self.ensemble.is_trained.values()):
            messagebox.showwarning("Attention", "Veuillez d'abord entraîner l'IA spécialisée")
            return

        if self.processed_data is None:
            messagebox.showwarning("Attention", "Aucune donnée disponible")
            return

        self.backtest_text.delete(1.0, tk.END)
        self.backtest_comparison_text.delete(1.0, tk.END)
        
        self.backtest_text.insert(tk.END, "🚀 Lancement du backtesting spécialisé...\n\n")

        # Lancer dans un thread
        thread = threading.Thread(target=self._run_specialized_backtest_thread)
        thread.daemon = True
        thread.start()

    def _run_specialized_backtest_thread(self):
        """Thread de backtesting spécialisé"""
        try:
            test_period = self.backtest_period_var.get() / 100
            strategy = self.strategy_var.get()
            test_type = self.backtest_type_var.get()

            # Configurer le moteur de backtesting avec l'ensemble spécialisé
            self.backtesting_engine.ensemble_model = self.ensemble

            if test_type == 'all':
                # Tester tous les types disponibles
                results_by_type = {}
                
                for race_type in ['GALOP', 'TROT']:
                    if self.ensemble.is_trained[race_type]:
                        # Filtrer les données par type
                        type_data = self.processed_data[self.processed_data['allure'] == race_type]
                        
                        if len(type_data) > 50:  # Minimum de données
                            try:
                                # Lancer le backtest pour ce type
                                results = self.backtesting_engine.run_backtest(
                                    data=type_data,
                                    strategy=strategy,
                                    test_period=test_period,
                                    feature_names=self.ensemble.feature_names[race_type]
                                )
                                results_by_type[race_type] = results
                                
                            except Exception as e:
                                self.queue.put(('backtest_error', f"Erreur backtest {race_type}: {str(e)}"))

                # Générer les rapports comparatifs
                if results_by_type:
                    global_report = self._generate_comparative_backtest_report(results_by_type)
                    self.queue.put(('backtest_results', global_report))
                    
                    comparison_report = self._generate_comparison_report(results_by_type)
                    self.queue.put(('backtest_comparison', comparison_report))

            else:
                # Tester un type spécifique
                if not self.ensemble.is_trained[test_type]:
                    self.queue.put(('backtest_error', f"Modèle {test_type} non entraîné"))
                    return

                type_data = self.processed_data[self.processed_data['allure'] == test_type]
                
                if len(type_data) < 50:
                    self.queue.put(('backtest_error', f"Pas assez de données {test_type}"))
                    return

                results = self.backtesting_engine.run_backtest(
                    data=type_data,
                    strategy=strategy,
                    test_period=test_period,
                    feature_names=self.ensemble.feature_names[test_type]
                )

                # Générer le rapport spécialisé
                report = self._generate_specialized_backtest_report(results, test_type)
                self.queue.put(('backtest_results', report))

        except Exception as e:
            self.queue.put(('backtest_error', f"Erreur lors du backtesting spécialisé: {str(e)}"))

    def _generate_comparative_backtest_report(self, results_by_type):
        """Générer un rapport comparatif de backtest"""
        report = "📈 RAPPORT DE BACKTESTING SPÉCIALISÉ\n"
        report += "=" * 70 + "\n\n"

        report += f"🎯 COMPARAISON DES PERFORMANCES PAR TYPE\n"
        report += "=" * 50 + "\n\n"

        for race_type, results in results_by_type.items():
            icon = "🏇" if race_type == 'GALOP' else "🐎"
            report += f"{icon} {race_type}\n"
            report += f"{'-'*20}\n"
            
            metrics = results.get('performance_metrics', {})
            
            report += f"📊 Courses testées: {results.get('total_races', 0)}\n"
            report += f"🎯 Précision: {metrics.get('accuracy', 0):.1f}%\n"
            report += f"✅ Taux réussite: {metrics.get('hit_rate', 0):.1f}%\n"
            report += f"💰 ROI moyen: {metrics.get('avg_roi_per_race', 0):.2f}%\n"
            report += f"⭐ Performance: {metrics.get('performance_rating', 'N/A')}\n"
            report += f"💹 Rentabilité: {metrics.get('profitability_rating', 'N/A')}\n\n"

        # Analyse comparative
        if len(results_by_type) >= 2:
            report += f"🏆 ANALYSE COMPARATIVE\n"
            report += f"{'='*30}\n"
            
            best_accuracy = max(results_by_type.items(), 
                               key=lambda x: x[1].get('performance_metrics', {}).get('accuracy', 0))
            best_roi = max(results_by_type.items(), 
                          key=lambda x: x[1].get('performance_metrics', {}).get('avg_roi_per_race', -100))
            
            report += f"🎯 Meilleure précision: {best_accuracy[0]} ({best_accuracy[1].get('performance_metrics', {}).get('accuracy', 0):.1f}%)\n"
            report += f"💰 Meilleur ROI: {best_roi[0]} ({best_roi[1].get('performance_metrics', {}).get('avg_roi_per_race', 0):.2f}%)\n\n"

        return report

    def _generate_comparison_report(self, results_by_type):
        """Générer un rapport de comparaison détaillé"""
        report = "⚖️ COMPARAISON DÉTAILLÉE GALOP vs TROT\n"
        report += "=" * 60 + "\n\n"

        if 'GALOP' in results_by_type and 'TROT' in results_by_type:
            galop_results = results_by_type['GALOP']
            trot_results = results_by_type['TROT']
            
            galop_metrics = galop_results.get('performance_metrics', {})
            trot_metrics = trot_results.get('performance_metrics', {})

            # Tableau comparatif
            report += f"{'Métrique':<20} {'GALOP':<15} {'TROT':<15} {'Avantage':<15}\n"
            report += f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}\n"

            metrics_to_compare = [
                ('Précision (%)', 'accuracy'),
                ('Taux réussite (%)', 'hit_rate'),
                ('ROI moyen (%)', 'avg_roi_per_race'),
                ('Courses testées', None)
            ]

            for metric_name, metric_key in metrics_to_compare:
                if metric_key:
                    galop_val = galop_metrics.get(metric_key, 0)
                    trot_val = trot_metrics.get(metric_key, 0)
                else:  # Courses testées
                    galop_val = galop_results.get('total_races', 0)
                    trot_val = trot_results.get('total_races', 0)

                if galop_val > trot_val:
                    advantage = "🏇 GALOP"
                elif trot_val > galop_val:
                    advantage = "🐎 TROT"
                else:
                    advantage = "⚖️ Égalité"

                report += f"{metric_name:<20} {galop_val:<15.1f} {trot_val:<15.1f} {advantage:<15}\n"

            # Conclusions
            report += f"\n📋 CONCLUSIONS:\n"
            report += f"{'-'*20}\n"
            
            # Analyser les forces de chaque type
            galop_strengths = []
            trot_strengths = []
            
            if galop_metrics.get('accuracy', 0) > trot_metrics.get('accuracy', 0):
                galop_strengths.append("Précision supérieure")
            else:
                trot_strengths.append("Précision supérieure")
                
            if galop_metrics.get('avg_roi_per_race', 0) > trot_metrics.get('avg_roi_per_race', 0):
                galop_strengths.append("ROI supérieur")
            else:
                trot_strengths.append("ROI supérieur")

            if galop_strengths:
                report += f"🏇 Forces GALOP: {', '.join(galop_strengths)}\n"
            if trot_strengths:
                report += f"🐎 Forces TROT: {', '.join(trot_strengths)}\n"

            # Recommandations
            report += f"\n💡 RECOMMANDATIONS:\n"
            best_overall = 'GALOP' if galop_metrics.get('accuracy', 0) > trot_metrics.get('accuracy', 0) else 'TROT'
            report += f"• Pour la précision: Privilégier les modèles {best_overall}\n"
            
            best_roi = 'GALOP' if galop_metrics.get('avg_roi_per_race', 0) > trot_metrics.get('avg_roi_per_race', 0) else 'TROT'
            report += f"• Pour la rentabilité: Privilégier les modèles {best_roi}\n"

        else:
            report += "Données insuffisantes pour la comparaison\n"

        return report

    def _generate_specialized_backtest_report(self, results, race_type):
        """Générer un rapport de backtest spécialisé"""
        icon = "🏇" if race_type == 'GALOP' else "🐎"
        
        report = f"📈 RAPPORT DE BACKTESTING {race_type} {icon}\n"
        report += "=" * 60 + "\n\n"

        # Informations générales
        report += f"🎯 Type de course: {race_type}\n"
        report += f"📊 Période de test: {results.get('test_period', 0)*100:.0f}%\n"
        report += f"🏁 Courses testées: {results.get('total_races', 0)}\n"
        report += f"📅 Date: {results.get('timestamp', 'N/A')[:10]}\n\n"

        # Métriques de performance spécialisées
        metrics = results.get('performance_metrics', {})

        report += f"📊 MÉTRIQUES {race_type}\n"
        report += "-" * 40 + "\n"
        report += f"🎯 Précision prédictions: {metrics.get('accuracy', 0):.1f}%\n"
        report += f"✅ Taux réussite paris: {metrics.get('hit_rate', 0):.1f}%\n"
        report += f"💰 ROI moyen/course: {metrics.get('avg_roi_per_race', 0):.2f}%\n"
        report += f"💎 ROI total: {metrics.get('total_roi', 0):.2f}%\n"
        report += f"🎲 Total paris: {results.get('total_bets', 0)}\n"
        report += f"🏆 Paris gagnants: {results.get('winning_bets', 0)}\n\n"

        # Évaluation spécialisée
        report += f"⭐ Performance {race_type}: {metrics.get('performance_rating', 'N/A')}\n"
        report += f"💹 Rentabilité {race_type}: {metrics.get('profitability_rating', 'N/A')}\n\n"

        # Conseils spécialisés
        report += f"💡 CONSEILS SPÉCIALISÉS {race_type}:\n"
        report += "-" * 40 + "\n"
        
        if race_type == 'GALOP':
            report += f"• Le galop montre une variabilité plus élevée\n"
            report += f"• Attention aux conditions de piste et météo\n"
            report += f"• Les poids et handicaps sont cruciaux\n"
            report += f"• Surveiller la forme récente et les œillères\n"
        elif race_type == 'TROT':
            report += f"• Le trot est plus prévisible chronométriquement\n"
            report += f"• Les temps précédents sont très indicatifs\n"
            report += f"• L'avis entraîneur a un poids important\n"
            report += f"• Attention au ferrage et technique de course\n"

        # Statistiques place si disponibles
        place_stats = results.get('place_stats', {})
        if place_stats.get('first_choice', {}).get('total', 0) > 0:
            report += f"\n🥇 STATISTIQUES PREMIER CHOIX {race_type}\n"
            report += "-" * 40 + "\n"
            first = place_stats['first_choice']
            report += f"• Victoires: {first['wins']}/{first['total']} ({metrics.get('first_choice_win_rate', 0):.1f}%)\n"
            report += f"• Top 3: {first['top3']}/{first['total']} ({metrics.get('first_choice_place_rate', 0):.1f}%)\n"

        return report

    # ===== MÉTHODES DE SAUVEGARDE/CHARGEMENT SPÉCIALISÉES =====

    def save_specialized_models(self):
        """Sauvegarder l'ensemble de modèles spécialisés"""
        if not any(self.ensemble.is_trained.values()):
            messagebox.showerror("Erreur", "Aucun modèle spécialisé entraîné à sauvegarder")
            return

        filename = filedialog.asksaveasfilename(
            title="Sauvegarder l'IA Spécialisée Galop/Trot",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if filename:
            try:
                save_data = {
                    'specialized_ensemble': self.ensemble,
                    'enhanced_feature_engineer': self.feature_engineer,
                    'training_results': self.training_results,
                    'race_type_stats': self.race_type_stats,
                    'version': '3.0_specialized',
                    'timestamp': datetime.now().isoformat(),
                    'specialization_info': {
                        'race_types_trained': [rt for rt, trained in self.ensemble.is_trained.items() if trained],
                        'total_specialized_models': self.ensemble.get_training_summary()['total_models']
                    }
                }

                with open(filename, 'wb') as f:
                    pickle.dump(save_data, f)

                # Résumé de sauvegarde
                summary = self.ensemble.get_training_summary()
                messagebox.showinfo("Succès",
                    f"IA Spécialisée sauvegardée avec succès !\n\n"
                    f"Contenu:\n"
                    f"• Types entraînés: {', '.join(summary['race_types_trained'])}\n"
                    f"• Modèles totaux: {summary['total_models']}\n"
                    f"• Features spécialisées par type\n"
                    f"• Résultats d'entraînement complets\n"
                    f"• Statistiques par type de course"
                )

            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur de sauvegarde spécialisée: {str(e)}")

    def load_specialized_models(self):
        """Charger l'ensemble de modèles spécialisés"""
        filename = filedialog.askopenfilename(
            title="Charger l'IA Spécialisée Galop/Trot",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'rb') as f:
                    save_data = pickle.load(f)

                # Vérifier la compatibilité spécialisée
                if 'version' in save_data and save_data['version'] == '3.0_specialized':
                    self.ensemble = save_data['specialized_ensemble']
                    self.feature_engineer = save_data.get('enhanced_feature_engineer', self.feature_engineer)
                    self.training_results = save_data.get('training_results', {})
                    self.race_type_stats = save_data.get('race_type_stats', {})

                    # Mettre à jour le moteur de backtesting
                    self.backtesting_engine.ensemble_model = self.ensemble

                    # Informations de spécialisation
                    spec_info = save_data.get('specialization_info', {})
                    
                    messagebox.showinfo("Succès",
                        f"IA Spécialisée chargée avec succès !\n\n"
                        f"Contenu:\n"
                        f"• Types: {', '.join(spec_info.get('race_types_trained', []))}\n"
                        f"• Modèles: {spec_info.get('total_specialized_models', 0)}\n"
                        f"• Features spécialisées chargées\n"
                        f"• Sauvegardé: {save_data.get('timestamp', 'Date inconnue')[:10]}"
                    )

                    # Activer les boutons
                    self.save_btn.config(state='normal')

                    # Afficher les résultats dans les onglets spécialisés
                    self._display_loaded_training_results()

                else:
                    messagebox.showwarning("Attention",
                        "Format de fichier non compatible avec la version spécialisée.\n"
                        "Veuillez utiliser un fichier sauvegardé avec l'IA Spécialisée."
                    )

            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur de chargement spécialisé: {str(e)}")

    def _display_loaded_training_results(self):
        """Afficher les résultats d'entraînement chargés"""
        if not self.training_results:
            return

        # Afficher dans l'onglet GALOP
        if 'GALOP' in self.training_results:
            galop_text = "🏇 IA GALOP CHARGÉE\n" + "="*40 + "\n\n"
            for target, results in self.training_results['GALOP'].items():
                galop_text += f"📊 {target.upper()}:\n"
                for model, score in results.items():
                    galop_text += f"  • {model}: {score:.4f}\n"
                galop_text += "\n"

            self.galop_results_text.delete(1.0, tk.END)
            self.galop_results_text.insert(1.0, galop_text)

        # Afficher dans l'onglet TROT
        if 'TROT' in self.training_results:
            trot_text = "🐎 IA TROT CHARGÉE\n" + "="*40 + "\n\n"
            for target, results in self.training_results['TROT'].items():
                trot_text += f"📊 {target.upper()}:\n"
                for model, score in results.items():
                    trot_text += f"  • {model}: {score:.4f}\n"
                trot_text += "\n"

            self.trot_results_text.delete(1.0, tk.END)
            self.trot_results_text.insert(1.0, trot_text)

        # Résumé global
        summary = self.ensemble.get_training_summary()
        global_text = "📊 RÉSUMÉ IA SPÉCIALISÉE CHARGÉE\n" + "="*50 + "\n\n"
        global_text += f"Types entraînés: {', '.join(summary['race_types_trained'])}\n"
        global_text += f"Modèles totaux: {summary['total_models']}\n\n"

        for race_type, info in summary['performance_by_type'].items():
            icon = "🏇" if race_type == 'GALOP' else "🐎"
            global_text += f"{icon} {race_type}: {info['model_count']} modèles\n"
            for target, score in info['performance'].items():
                global_text += f"  • {target}: {score:.4f}\n"
            global_text += "\n"

        self.global_results_text.delete(1.0, tk.END)
        self.global_results_text.insert(1.0, global_text)

    # ===== GESTION DES ÉVÉNEMENTS COMPLÈTE =====

    def check_queue(self):
        """Vérifier la queue pour les mises à jour des threads - Version spécialisée"""
        try:
            while True:
                msg_type, *args = self.queue.get_nowait()

                if msg_type == 'progress':
                    progress, label = args
                    self.progress_var.set(progress)
                    self.progress_label.config(text=label)
                    self.status_var.set(label)

                elif msg_type == 'specialized_stats':
                    race_type, horses, races, avg, features = args
                    if race_type == 'GALOP':
                        self.galop_stats['horses'].set(f"{horses:,}")
                        self.galop_stats['races'].set(f"{races:,}")
                        self.galop_stats['avg'].set(f"{avg:.1f}")
                        self.galop_stats['features'].set(f"{features}")
                    elif race_type == 'TROT':
                        self.trot_stats['horses'].set(f"{horses:,}")
                        self.trot_stats['races'].set(f"{races:,}")
                        self.trot_stats['avg'].set(f"{avg:.1f}")
                        self.trot_stats['features'].set(f"{features}")

                elif msg_type == 'complete':
                    message = args[0]
                    messagebox.showinfo("Succès", message)
                    self.process_btn.config(state='normal')
                    self.train_btn.config(state='normal')

                    # Remplir la liste des courses avec détection de type
                    if self.processed_data is not None:
                        race_files = sorted(self.processed_data['race_file'].unique())
                        self.race_combo['values'] = race_files
                        if race_files:
                            self.race_combo.set(race_files[0])
                            self.on_race_selection_change()

                elif msg_type == 'error':
                    message = args[0]
                    messagebox.showerror("Erreur", message)
                    self.process_btn.config(state='normal')

                elif msg_type == 'warning':
                    message = args[0]
                    print(f"Warning: {message}")

                elif msg_type == 'training_info':
                    target_tab, message = args
                    if target_tab == 'galop':
                        self.galop_results_text.insert(tk.END, message)
                        self.galop_results_text.see(tk.END)
                    elif target_tab == 'trot':
                        self.trot_results_text.insert(tk.END, message)
                        self.trot_results_text.see(tk.END)
                    elif target_tab == 'global':
                        self.global_results_text.insert(tk.END, message)
                        self.global_results_text.see(tk.END)

                elif msg_type == 'specialized_training_complete':
                    message = args[0]
                    messagebox.showinfo("🎉 Succès Spécialisé", message)
                    self.train_btn.config(state='normal')
                    self.save_btn.config(state='normal')

                elif msg_type == 'training_error':
                    message = args[0]
                    messagebox.showerror("Erreur", message)
                    self.train_btn.config(state='normal')

                elif msg_type == 'backtest_results':
                    report = args[0]
                    self.backtest_text.delete(1.0, tk.END)
                    self.backtest_text.insert(1.0, report)

                elif msg_type == 'backtest_comparison':
                    report = args[0]
                    self.backtest_comparison_text.delete(1.0, tk.END)
                    self.backtest_comparison_text.insert(1.0, report)

                elif msg_type == 'backtest_error':
                    error_msg = args[0]
                    messagebox.showerror("Erreur Backtest Spécialisé", error_msg)
                    self.backtest_text.insert(tk.END, f"\n❌ ERREUR: {error_msg}\n")

        except queue.Empty:
            pass

        # Programmer la prochaine vérification
        self.root.after(100, self.check_queue)

    def show_welcome_message(self):
        """Afficher le message de bienvenue spécialisé"""
        welcome_msg = (
            "🎉 Bienvenue dans le Prédicteur Hippique PRO Spécialisé !\n\n"
            "🚀 NOUVEAUTÉS DE CETTE VERSION SPÉCIALISÉE :\n"
            "✅ IA Spécialisée avec modèles distincts pour Galop et Trot\n"
            "✅ Features dédiées : 50+ pour Galop, 55+ pour Trot\n"
            "✅ Détection automatique du type de course\n"
            "✅ Prédictions optimisées par discipline\n"
            "✅ Backtesting comparatif Galop vs Trot\n"
            "✅ Analytics spécialisés avec graphiques dédiés\n"
            "✅ Sauvegarde/chargement des modèles spécialisés\n\n"
            "🏇 GALOP - Features spécialisées :\n"
            "• Poids et handicaps (impact physique direct)\n"
            "• Œillères et équipement (concentration)\n"
            "• Qualité du finish (distance à l'arrivée)\n"
            "• Statut jument pleine (condition physique)\n\n"
            "🐎 TROT - Features spécialisées :\n"
            "• Performance chronométrée (temps/vitesse)\n"
            "• Technique de course (ferrage, avis entraîneur)\n"
            "• Handicap distance (placement départ)\n"
            "• Réduction kilométrique (performance relative)\n\n"
            "📁 POUR COMMENCER :\n"
            "1. Allez dans l'onglet 'Données Spécialisées'\n"
            "2. Chargez vos fichiers JSON (Galop + Trot)\n"
            "3. Observez la détection automatique des types\n"
            "4. Entraînez l'IA Spécialisée avec modèles séparés\n"
            "5. Profitez des prédictions optimisées par discipline !\n\n"
            "💡 CONSEIL : Chaque type de course a ses propres modèles pour une précision maximale."
        )
        
        messagebox.showinfo("🏇 Prédicteur Hippique PRO Spécialisé", welcome_msg)


def main():
    """Fonction principale de l'application spécialisée"""
    print("🏇 Lancement du Prédicteur Hippique PRO - VERSION SPÉCIALISÉE GALOP/TROT")
    print("=" * 80)
    print("🤖 IA Spécialisée: Modèles distincts pour Galop et Trot")
    print("🔧 Features dédiées: 50+ pour Galop, 55+ pour Trot")
    print("⚡ Optimisations: Détection automatique + Prédictions ciblées")
    print("📊 Analytics: Comparaisons Galop vs Trot + Backtesting spécialisé")
    print("💾 Sauvegarde: Modèles spécialisés avec métriques séparées")
    print("🎯 Prédictions: Recommandations optimisées par discipline")
    print("=" * 80)
    
    # Vérifier les dépendances (fonction existante)
    print("🔍 Vérification des dépendances...")
    try:
        import pandas as pd
        import numpy as np
        import lightgbm as lgb
        import xgboost as xgb
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        print("✅ Toutes les dépendances principales sont installées")
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("💡 Installez avec: pip install pandas numpy lightgbm xgboost matplotlib seaborn scikit-learn")
        sys.exit(1)
    
    # Configuration de l'environnement
    print("⚙️ Configuration de l'environnement spécialisé...")
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        print("✅ Environnement configuré")
    except:
        print("⚠️ Configuration partielle")
    
    # Création de l'interface spécialisée
    print("🎨 Initialisation de l'interface spécialisée...")
    try:
        root = tk.Tk()
        
        # Configuration de la fenêtre principale
        root.minsize(1200, 800)
        
        # Gestion de la fermeture propre
        def on_closing():
            """Gestion de la fermeture de l'application spécialisée"""
            if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter l'application spécialisée ?"):
                print("👋 Fermeture de l'application spécialisée...")
                try:
                    root.quit()
                    root.destroy()
                except:
                    pass
                print("✅ Application spécialisée fermée proprement")
                sys.exit(0)
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Création de l'application spécialisée
        print("🚀 Création de l'application spécialisée...")
        app = EnhancedHorseRacingGUI(root)
        print("✅ Application spécialisée créée avec succès")
        
        # Informations système
        try:
            if PSUTIL_AVAILABLE:
                memory_mb = psutil.Process().memory_info().rss / 1024**2
                print(f"💾 Utilisation mémoire initiale: {memory_mb:.0f}MB")
        except:
            print("💾 Monitoring mémoire non disponible")
        
        print("🎉 Interface spécialisée prête !")
        print("📋 Fonctionnalités spécialisées disponibles:")
        print("   • Chargement avec détection automatique Galop/Trot")
        print("   • Entraînement IA avec modèles séparés par discipline")
        print("   • Prédictions spécialisées avec features dédiées")
        print("   • Analytics comparatifs Galop vs Trot")
        print("   • Backtesting spécialisé avec rapports détaillés")
        print("   • Sauvegarde/chargement des modèles spécialisés")
        print("=" * 80)
        
        # Lancement de la boucle principale
        root.mainloop()
        
    except Exception as e:
        error_msg = (
            f"❌ Erreur critique lors du lancement spécialisé :\n"
            f"{str(e)}\n\n"
            f"Vérifications à effectuer :\n"
            f"1. Python 3.7+ installé\n"
            f"2. Toutes les dépendances installées\n"
            f"3. Modules spécialisés présents (enhanced_feature_engineer.py, etc.)\n"
            f"4. Interface graphique disponible\n"
            f"5. Permissions d'écriture dans le répertoire"
        )
        
        print(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()

    def show_specialized_features(self):
        """Afficher l'importance des features spécialisées"""
        # Placeholder pour l'importance des features spécialisées
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()

        info_label = ttk.Label(self.analysis_frame,
                              text="🔧 Analyse des Features Spécialisées\n\n" +
                                   "Cette fonctionnalité analysera :\n" +
                                   "• Importance des features Galop vs Trot\n" +
                                   "• Comparaison des features communes\n" +
                                   "• Impact des features spécialisées\n\n" +
                                   "Entraînez d'abord les modèles spécialisés.",
                              font=('Arial', 12), justify='center')
        info_label.pack(expand=True)

    def show_galop_trot_comparison(self):
        """Afficher la comparaison Galop vs Trot"""
        if self.processed_data is None:
            messagebox.showwarning("Attention", "Aucune donnée chargée")
            return

        # Nettoyer le frame
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparaison Galop vs Trot - Analyses Statistiques', fontsize=16, fontweight='bold')

        try:
            # Séparer les données par type
            galop_data = self.processed_data[self.processed_data['allure'] == 'GALOP']
            trot_data = self.processed_data[self.processed_data['allure'] == 'TROT']

            # Graphique 1: Comparaison des taux de victoire
            if not galop_data.empty and not trot_data.empty:
                galop_winrate = galop_data['win_rate'].mean()
                trot_winrate = trot_data['win_rate'].mean()
                
                types = ['GALOP', 'TROT']
                winrates = [galop_winrate, trot_winrate]
                colors = ['#e74c3c', '#3498db']
                
                bars = axes[0, 0].bar(types, winrates, color=colors, alpha=0.8)
                axes[0, 0].set_ylabel('Taux de Victoire Moyen')
                axes[0, 0].set_title('Comparaison Taux de Victoire')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Ajouter les valeurs sur les barres
                for bar, rate in zip(bars, winrates):
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                                   f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')

            # Graphique 2: Distribution des cotes
            if not galop_data.empty and not trot_data.empty:
                axes[0, 1].hist(galop_data['direct_odds'], bins=20, alpha=0.7, label='GALOP', 
                               color='#e74c3c', density=True)
                axes[0, 1].hist(trot_data['direct_odds'], bins=20, alpha=0.7, label='TROT', 
                               color='#3498db', density=True)
                axes[0, 1].set_xlabel('Cotes Directes')
                axes[0, 1].set_ylabel('Densité')
                axes[0, 1].set_title('Distribution des Cotes')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

            # Graphique 3: Expérience moyenne
            if not galop_data.empty and not trot_data.empty:
                galop_exp = galop_data['nombreCourses'].mean()
                trot_exp = trot_data['nombreCourses'].mean()
                
                types = ['GALOP', 'TROT']
                experiences = [galop_exp, trot_exp]
                
                bars = axes[1, 0].bar(types, experiences, color=colors, alpha=0.8)
                axes[1, 0].set_ylabel('Nombre Moyen de Courses')
                axes[1, 0].set_title('Expérience Moyenne')
                axes[1, 0].grid(True, alpha=0.3)
                
                for bar, exp in zip(bars, experiences):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                   f'{exp:.1f}', ha='center', va='bottom', fontweight='bold')

            # Graphique 4: Gains moyens
            if not galop_data.empty and not trot_data.empty:
                galop_gains = galop_data['gains_carriere'].mean()
                trot_gains = trot_data['gains_carriere'].mean()
                
                types = ['GALOP', 'TROT']
                gains = [galop_gains, trot_gains]
                
                bars = axes[1, 1].bar(types, gains, color=colors, alpha=0.8)
                axes[1, 1].set_ylabel('Gains Carrière Moyens (€)')
                axes[1, 1].set_title('Comparaison Gains')
                axes[1, 1].grid(True, alpha=0.3)
                
                for bar, gain in zip(bars, gains):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + gain*0.01,
                                   f'{gain:.0f}€', ha='center', va='bottom', fontweight='bold')

        except Exception as e:
            for ax in axes.flat:
                ax.text(0.5, 0.5, f'Erreur:\n{str(e)}',
                       ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.analysis_frame)
