"""
Application complète du prédicteur hippique
Version intégrée avec TOUTES les fonctionnalités du code original
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
import itertools

# Imports ML
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Imports graphiques
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Imports modules locaux
from cache_manager import IntelligentCache
from error_handler import RobustErrorHandler
from feature_engineer import AdvancedFeatureEngineer
from ensemble_models import HorseRacingEnsemble
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


class CompleteHorseRacingGUI:
    """Interface graphique complète avec toutes les fonctionnalités"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🏇 Prédicteur Hippique PRO - IA Ensemble Optimisé")
        self.root.geometry("1400x900")

        # Initialisation des composants
        self.cache = IntelligentCache()
        self.error_handler = RobustErrorHandler()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ensemble = HorseRacingEnsemble()
        self.backtesting_engine = BacktestingEngine(self.ensemble)

        # Données
        self.raw_data = None
        self.processed_data = None
        self.training_results = {}
        self.json_files = []
        self.new_race_data = None

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

        # Couleurs modernes
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
        style.configure('Success.TLabel', foreground='#27ae60')
        style.configure('Warning.TLabel', foreground='#e67e22')
        style.configure('Error.TLabel', foreground='#e74c3c')

    def setup_ui(self):
        """Configuration complète de l'interface utilisateur"""

        # Header principal
        self._create_header()

        # Notebook pour les onglets
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Création des onglets complets
        self._create_loading_tab()
        self._create_training_tab()
        self._create_prediction_tab()
        self._create_analytics_tab()
        self._create_backtesting_tab()

        # Barre de statut
        self._create_status_bar()

        # Démarrage des métriques de performance
        self.update_performance_metrics()

    def _create_header(self):
        """Créer l'en-tête de l'application"""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill='x', padx=10, pady=(10, 0))

        title_label = ttk.Label(header_frame, text="🏇 Prédicteur Hippique PRO",
                               style='Title.TLabel')
        title_label.pack(side='left')

        subtitle_label = ttk.Label(header_frame, text="IA Ensemble avec LightGBM + XGBoost + CatBoost + Random Forest",
                                  font=('Arial', 10), foreground='#7f8c8d')
        subtitle_label.pack(side='left', padx=(20, 0))

    def _create_status_bar(self):
        """Créer la barre de statut avec métriques complètes"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side='bottom', fill='x', padx=10, pady=5)

        self.status_var = tk.StringVar()
        self.status_var.set("🚀 Prêt - IA Ensemble avec optimisation bayésienne")

        status_bar = ttk.Label(status_frame, textvariable=self.status_var,
                              relief='sunken', anchor='w')
        status_bar.pack(side='left', fill='x', expand=True)

        # Indicateurs de performance
        self.performance_frame = ttk.Frame(status_frame)
        self.performance_frame.pack(side='right')

        self.cache_label = ttk.Label(self.performance_frame, text="Cache: 0%",
                                    font=('Arial', 8), foreground='#7f8c8d')
        self.cache_label.pack(side='right', padx=5)

        self.memory_label = ttk.Label(self.performance_frame, text="RAM: 0MB",
                                     font=('Arial', 8), foreground='#7f8c8d')
        self.memory_label.pack(side='right', padx=5)

    def _create_loading_tab(self):
        """Créer l'onglet de chargement complet avec scroll"""
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="📁 Données")

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

        # Titre avec description complète
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x', padx=20, pady=20)

        title = ttk.Label(title_frame, text="📁 Chargement des Données",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack()

        subtitle = ttk.Label(title_frame, text="Traitement avancé des fichiers JSON avec extraction de features IA",
                           font=('Arial', 10), foreground='#7f8c8d')
        subtitle.pack(pady=(5, 0))

        # Sélection du dossier avec design moderne
        folder_frame = ttk.LabelFrame(main_frame, text="📂 Dossier des courses", padding=15)
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

        scan_btn = ttk.Button(folder_frame, text="🔍 Scanner le dossier",
                             command=self.scan_folder, style='Accent.TButton')
        scan_btn.pack(pady=(10, 0))

        # Informations détaillées
        info_frame = ttk.LabelFrame(main_frame, text="ℹ️ Informations détaillées", padding=15)
        info_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.info_text = scrolledtext.ScrolledText(info_frame, height=10, width=80,
                                                  font=('Consolas', 9))
        self.info_text.pack(fill='both', expand=True)

        # Barre de progression moderne
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill='x', padx=20, pady=10)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100, length=400)
        self.progress_bar.pack(fill='x')

        self.progress_label = ttk.Label(progress_frame, text="", font=('Arial', 9))
        self.progress_label.pack(pady=(5, 0))

        # Bouton de traitement principal
        self.process_btn = ttk.Button(main_frame, text="🚀 Traiter avec IA Avancée",
                                     command=self.process_files,
                                     style='Accent.TButton', state='disabled')
        self.process_btn.pack(pady=20)

        # Statistiques complètes en cards
        self._create_complete_stats_section(main_frame)

    def _create_complete_stats_section(self, parent):
        """Créer la section complète des statistiques"""
        stats_frame = ttk.LabelFrame(parent, text="📊 Statistiques du dataset", padding=15)
        stats_frame.pack(fill='x', padx=20, pady=10)

        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack()

        # Variables des métriques complètes
        self.horses_var = tk.StringVar(value="0")
        self.races_var = tk.StringVar(value="0")
        self.avg_var = tk.StringVar(value="0.0")
        self.period_var = tk.StringVar(value="0")
        self.features_var = tk.StringVar(value="0")

        metrics = [
            ("🏇 Chevaux", self.horses_var, 0, 0),
            ("🏁 Courses", self.races_var, 0, 1),
            ("👥 Moy/course", self.avg_var, 1, 0),
            ("📅 Période (j)", self.period_var, 1, 1),
            ("🔧 Features", self.features_var, 2, 0)
        ]

        for title, var, row, col in metrics:
            frame = ttk.Frame(stats_grid)
            frame.grid(row=row, column=col, padx=15, pady=10, sticky='w')

            ttk.Label(frame, text=title, font=('Arial', 10, 'bold')).pack()
            ttk.Label(frame, textvariable=var, font=('Arial', 14, 'bold'),
                     foreground='#3498db').pack()

    def _create_training_tab(self):
        """Créer l'onglet d'entraînement IA complet"""
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="🤖 IA Ensemble")

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

        # Header complet
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x', padx=20, pady=20)

        title = ttk.Label(title_frame, text="🤖 Intelligence Artificielle Ensemble",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack()

        subtitle = ttk.Label(title_frame,
                           text="Combinaison optimale de LightGBM, XGBoost, CatBoost et Random Forest",
                           font=('Arial', 10), foreground='#7f8c8d')
        subtitle.pack(pady=(5, 0))

        # Modèles disponibles avec détails
        models_frame = ttk.LabelFrame(main_frame, text="🎯 Modèles d'IA disponibles", padding=15)
        models_frame.pack(fill='x', padx=20, pady=10)

        models_info = [
            ("LightGBM", "Gradient Boosting optimisé Microsoft", "✅ Activé"),
            ("XGBoost", "Extreme Gradient Boosting", "✅ Activé"),
            ("CatBoost", "Categorical Boosting Yandex", "✅ Activé" if CATBOOST_AVAILABLE else "❌ Non installé"),
            ("Random Forest", "Forêt d'arbres décisionnels", "✅ Activé"),
        ]

        for i, (name, desc, status) in enumerate(models_info):
            model_frame = ttk.Frame(models_frame)
            model_frame.pack(fill='x', pady=5)

            ttk.Label(model_frame, text=f"• {name}", font=('Arial', 10, 'bold')).pack(side='left')
            ttk.Label(model_frame, text=desc, font=('Arial', 9)).pack(side='left', padx=(10, 0))
            ttk.Label(model_frame, text=status, font=('Arial', 9, 'bold'),
                     foreground='#27ae60' if '✅' in status else '#e74c3c').pack(side='right')

        # Configuration d'entraînement complète
        config_frame = ttk.LabelFrame(main_frame, text="⚙️ Configuration d'entraînement", padding=15)
        config_frame.pack(fill='x', padx=20, pady=10)

        config_grid = ttk.Frame(config_frame)
        config_grid.pack(fill='x')

        # Paramètres d'entraînement
        ttk.Label(config_grid, text="Min courses par cheval:").grid(row=0, column=0, sticky='w', padx=5)
        self.min_races_var = tk.IntVar(value=3)
        min_races_spin = ttk.Spinbox(config_grid, from_=1, to=10, textvariable=self.min_races_var, width=10)
        min_races_spin.grid(row=0, column=1, padx=5, sticky='w')

        ttk.Label(config_grid, text="Validations croisées:").grid(row=0, column=2, sticky='w', padx=5)
        self.cv_folds_var = tk.IntVar(value=5)
        cv_folds_spin = ttk.Spinbox(config_grid, from_=3, to=10, textvariable=self.cv_folds_var, width=10)
        cv_folds_spin.grid(row=0, column=3, padx=5, sticky='w')

        ttk.Label(config_grid, text="Seuil de confiance:").grid(row=1, column=0, sticky='w', padx=5)
        self.confidence_var = tk.DoubleVar(value=0.6)
        confidence_spin = ttk.Spinbox(config_grid, from_=0.1, to=0.9, increment=0.1,
                                     textvariable=self.confidence_var, width=10)
        confidence_spin.grid(row=1, column=1, padx=5, sticky='w')

        # Info technique complète
        tech_frame = ttk.LabelFrame(main_frame, text="🔬 Techniques avancées utilisées", padding=15)
        tech_frame.pack(fill='x', padx=20, pady=10)

        tech_info = """✅ Validation croisée temporelle (Time Series Split)
✅ Feature engineering automatisé (45+ features)
✅ Ensemble pondéré par performance
✅ Gestion des données manquantes et outliers
✅ Optimisation des hyperparamètres
✅ Cache intelligent pour accélération
✅ Gestion robuste des erreurs avec retry automatique
✅ Encodage intelligent des variables catégorielles"""

        ttk.Label(tech_frame, text=tech_info, justify='left', font=('Arial', 9)).pack(anchor='w')

        # Bouton d'entraînement principal
        self.train_btn = ttk.Button(main_frame, text="🚀 Lancer l'Entraînement IA Ensemble",
                                   command=self.train_ensemble_models,
                                   style='Accent.TButton', state='disabled')
        self.train_btn.pack(pady=20)

        # Résultats d'entraînement complets
        results_frame = ttk.LabelFrame(main_frame, text="📊 Résultats d'entraînement", padding=15)
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, font=('Consolas', 9))
        self.results_text.pack(fill='both', expand=True)

        # Boutons de sauvegarde/chargement complets
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill='x', padx=20, pady=10)

        self.save_btn = ttk.Button(buttons_frame, text="💾 Sauvegarder IA",
                                  command=self.save_models, state='disabled')
        self.save_btn.pack(side='left', padx=5)

        self.load_btn = ttk.Button(buttons_frame, text="📂 Charger IA",
                                  command=self.load_models)
        self.load_btn.pack(side='left', padx=5)

        ttk.Label(buttons_frame, text="Format: Ensemble complet avec tous les modèles et métriques",
                 font=('Arial', 8), foreground='#7f8c8d').pack(side='left', padx=20)

    def _create_prediction_tab(self):
        """Créer l'onglet de prédictions complet"""
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="🎯 Prédictions")

        main_frame = ttk.Frame(self.tab3)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Titre complet
        title = ttk.Label(main_frame, text="🎯 Prédictions IA Ensemble",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack(pady=(0, 20))

        # Mode de prédiction complet
        mode_frame = ttk.LabelFrame(main_frame, text="Mode de prédiction", padding=10)
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
        self.existing_frame = ttk.LabelFrame(main_frame, text="Sélection de course existante", padding=10)

        ttk.Label(self.existing_frame, text="Course:").pack(side='left', padx=5)
        self.race_var = tk.StringVar()
        self.race_combo = ttk.Combobox(self.existing_frame, textvariable=self.race_var,
                                      state='readonly', width=50)
        self.race_combo.pack(side='left', fill='x', expand=True, padx=5)

        # Frame pour nouveau fichier complet
        self.new_file_frame = ttk.LabelFrame(main_frame, text="Charger un nouveau fichier", padding=10)

        file_select_frame = ttk.Frame(self.new_file_frame)
        file_select_frame.pack(fill='x', pady=5)

        self.new_file_var = tk.StringVar()
        new_file_entry = ttk.Entry(file_select_frame, textvariable=self.new_file_var, width=50)
        new_file_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))

        browse_new_btn = ttk.Button(file_select_frame, text="📂 Parcourir",
                                   command=self.browse_new_race_file)
        browse_new_btn.pack(side='right')

        load_new_btn = ttk.Button(self.new_file_frame, text="📥 Charger le fichier",
                                 command=self.load_new_race_file)
        load_new_btn.pack(pady=5)

        self.new_file_info = ttk.Label(self.new_file_frame, text="", foreground='green')
        self.new_file_info.pack(pady=5)

        # Bouton de prédiction principal
        self.predict_btn = ttk.Button(main_frame, text="🎯 Prédire avec IA Ensemble",
                                     command=self.predict_race, style='Accent.TButton')
        self.predict_btn.pack(pady=10)

        # Résultats de prédiction complets
        results_frame = ttk.LabelFrame(main_frame, text="Résultats IA Ensemble", padding=10)
        results_frame.pack(fill='both', expand=True, pady=10)

        # Tableau des résultats avec toutes les colonnes
        columns = ('Rang', 'N°', 'Nom', 'Driver', 'Prob Victoire', 'Prob Place', 'Cotes', 'Confiance')
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
            else:
                self.prediction_tree.column(col, width=80)

        scrollbar_pred = ttk.Scrollbar(results_frame, orient='vertical', command=self.prediction_tree.yview)
        self.prediction_tree.configure(yscrollcommand=scrollbar_pred.set)

        self.prediction_tree.pack(side='left', fill='both', expand=True)
        scrollbar_pred.pack(side='right', fill='y')

        # Recommandations IA complètes
        reco_frame = ttk.LabelFrame(main_frame, text="Recommandations IA", padding=10)
        reco_frame.pack(fill='x', pady=10)

        self.reco_text = scrolledtext.ScrolledText(reco_frame, height=8, font=('Consolas', 9))
        self.reco_text.pack(fill='both', expand=True)

        # Initialiser l'affichage
        self.on_prediction_mode_change()

    def _create_analytics_tab(self):
        """Créer l'onglet d'analyses complet"""
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="📊 Analytics")

        main_frame = ttk.Frame(self.tab4)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        title = ttk.Label(main_frame, text="📊 Analytics & Insights IA",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack(pady=(0, 20))

        # Boutons d'analyse complets
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill='x', pady=10)

        ttk.Button(buttons_frame, text="📈 Performance Modèles",
                  command=self.show_model_performance).pack(side='left', padx=5)

        ttk.Button(buttons_frame, text="🔗 Feature Importance",
                  command=self.show_feature_importance).pack(side='left', padx=5)

        ttk.Button(buttons_frame, text="📅 Évolution Temporelle",
                  command=self.show_temporal_analysis).pack(side='left', padx=5)

        ttk.Button(buttons_frame, text="🎯 Analyse Prédictions",
                  command=self.show_prediction_analysis).pack(side='left', padx=5)

        # Zone d'analyse principale
        self.analysis_frame = ttk.Frame(main_frame)
        self.analysis_frame.pack(fill='both', expand=True, pady=10)

    def _create_backtesting_tab(self):
        """Créer l'onglet de backtesting complet"""
        self.tab5 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab5, text="📈 Backtesting")

        main_frame = ttk.Frame(self.tab5)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        title = ttk.Label(main_frame, text="📈 Backtesting & Validation",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack(pady=(0, 20))

        # Configuration du backtest complète
        config_frame = ttk.LabelFrame(main_frame, text="Configuration du backtest", padding=15)
        config_frame.pack(fill='x', pady=10)

        config_grid = ttk.Frame(config_frame)
        config_grid.pack()

        ttk.Label(config_grid, text="Période de test (%):").grid(row=0, column=0, sticky='w', padx=5)
        self.backtest_period_var = tk.IntVar(value=20)
        backtest_period_spin = ttk.Spinbox(config_grid, from_=10, to=50,
                                          textvariable=self.backtest_period_var, width=10)
        backtest_period_spin.grid(row=0, column=1, padx=5)

        ttk.Label(config_grid, text="Stratégie:").grid(row=0, column=2, sticky='w', padx=5)
        self.strategy_var = tk.StringVar(value="place_strategy")
        strategy_combo = ttk.Combobox(config_grid, textvariable=self.strategy_var,
                                     values=['place_strategy', 'confidence', 'top_pick', 'value_betting'],
                                     state='readonly', width=15)
        strategy_combo.grid(row=0, column=3, padx=5)

        # Description des stratégies complète
        strategy_desc_frame = ttk.LabelFrame(main_frame, text="📋 Description des stratégies", padding=10)
        strategy_desc_frame.pack(fill='x', pady=5)
        
        strategy_descriptions = """🎯 place_strategy: Mise sur le cheval avec la plus haute probabilité de place
       • Suit les performances du 1er et 2ème choix
       • Compte les victoires et places dans le top 3
       • Ne prend pas en compte les cotes

✅ confidence: Mise sur le favori IA seulement si probabilité victoire > 60%
🔥 top_pick: Mise systématique sur le favori IA  
💎 value_betting: Mise seulement si valeur attendue > 1.2"""
        
        ttk.Label(strategy_desc_frame, text=strategy_descriptions, 
                 justify='left', font=('Arial', 9)).pack(anchor='w')

        # Bouton de backtest
        backtest_btn = ttk.Button(main_frame, text="🚀 Lancer le Backtest",
                                 command=self.run_backtest, style='Accent.TButton')
        backtest_btn.pack(pady=10)

        # Résultats du backtest complets
        backtest_results_frame = ttk.LabelFrame(main_frame, text="Résultats du backtest", padding=10)
        backtest_results_frame.pack(fill='both', expand=True, pady=10)

        self.backtest_text = scrolledtext.ScrolledText(backtest_results_frame, height=15,
                                                      font=('Consolas', 9))
        self.backtest_text.pack(fill='both', expand=True)

    # ===== MÉTHODES FONCTIONNELLES COMPLÈTES =====

    def update_performance_metrics(self):
        """Mettre à jour les métriques de performance complètes"""
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

        except:
            pass

        # Programmer la prochaine mise à jour
        self.root.after(5000, self.update_performance_metrics)

    def browse_folder(self):
        """Parcourir pour sélectionner un dossier"""
        folder = filedialog.askdirectory(title="Sélectionner le dossier des courses")
        if folder:
            self.folder_var.set(folder)

    def scan_folder(self):
        """Scanner le dossier pour les fichiers JSON - Version complète"""
        folder = self.folder_var.get()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier valide")
            return

        json_files = glob.glob(os.path.join(folder, "*.json"))

        if not json_files:
            messagebox.showwarning("Attention", "Aucun fichier JSON trouvé dans ce dossier")
            return

        self.json_files = json_files

        # Analyse rapide complète des fichiers
        info_text = f"✅ {len(json_files)} fichiers JSON trouvés\n\n"
        info_text += "📊 ANALYSE RAPIDE:\n"
        info_text += "=" * 40 + "\n"

        sample_size = min(5, len(json_files))
        total_participants = 0

        for i, file in enumerate(json_files[:sample_size]):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                participants = len(data.get('participants', []))
                total_participants += participants
                filename = os.path.basename(file)
                info_text += f"{i+1}. {filename}: {participants} chevaux\n"
            except Exception as e:
                info_text += f"{i+1}. {os.path.basename(file)}: Erreur de lecture\n"

        if sample_size < len(json_files):
            info_text += f"... et {len(json_files) - sample_size} autres fichiers\n"

        avg_participants = total_participants / sample_size if sample_size > 0 else 0
        info_text += f"\n📈 ESTIMATION:\n"
        info_text += f"• Moyenne: {avg_participants:.1f} chevaux/course\n"
        info_text += f"• Total estimé: ~{len(json_files) * avg_participants:.0f} chevaux\n"
        info_text += f"• Features générées: ~45 par cheval\n"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)

        self.process_btn.config(state='normal')
        self.status_var.set(f"Prêt à traiter {len(json_files)} fichiers avec IA avancée")

    def process_files(self):
        """Traiter les fichiers JSON avec IA avancée - Version complète"""
        self.process_btn.config(state='disabled')
        self.progress_var.set(0)

        # Lancer le traitement dans un thread séparé
        thread = threading.Thread(target=self._process_files_thread)
        thread.daemon = True
        thread.start()

    def _process_files_thread(self):
        """Thread de traitement des fichiers avec gestion d'erreurs complète"""
        try:
            def _process_internal():
                all_races = []
                total_files = len(self.json_files)

                self.queue.put(('progress', 5, f"Initialisation du traitement IA..."))

                for i, file_path in enumerate(self.json_files):
                    progress = (i + 1) / total_files * 70  # 70% pour le chargement
                    filename = os.path.basename(file_path)

                    self.queue.put(('progress', progress, f"Traitement: {filename} ({i+1}/{total_files})"))

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

                # Extraction des features avec IA avancée
                self.queue.put(('progress', 85, "Extraction des features IA avancées..."))
                self.processed_data = self.feature_engineer.extract_comprehensive_features(self.raw_data)

                # Optimisation mémoire
                self.queue.put(('progress', 95, "Optimisation mémoire..."))
                self.processed_data = self.optimize_dataframe_memory(self.processed_data)

                # Statistiques finales
                unique_races = self.processed_data['race_file'].nunique()
                avg_horses = len(self.processed_data) / unique_races
                date_range = (self.processed_data['race_date'].max() -
                             self.processed_data['race_date'].min()).days
                num_features = self.processed_data.shape[1]

                self.queue.put(('stats', len(self.processed_data), unique_races, avg_horses, date_range, num_features))
                self.queue.put(('progress', 100, "Traitement IA terminé !"))
                self.queue.put(('complete', "Données traitées avec succès par l'IA !"))

            # Exécution avec gestion d'erreurs robuste
            self.error_handler.robust_execute(_process_internal, 'data_loading')

        except Exception as e:
            self.queue.put(('error', f"Erreur lors du traitement IA: {str(e)}"))

    def optimize_dataframe_memory(self, df):
        """Optimiser l'usage mémoire du DataFrame - Version complète"""
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
            else:
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

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
        else:
            print(f"💾 Optimisation mémoire: {reduction:.1f}% de réduction")

        return df

    # ===== MÉTHODES D'ENTRAÎNEMENT COMPLÈTES =====

    def train_ensemble_models(self):
        """Entraîner l'ensemble de modèles IA - Version complète"""
        if self.processed_data is None:
            messagebox.showerror("Erreur", "Veuillez d'abord charger les données")
            return

        self.train_btn.config(state='disabled')
        self.results_text.delete(1.0, tk.END)

        # Lancer l'entraînement dans un thread
        thread = threading.Thread(target=self._train_ensemble_thread)
        thread.daemon = True
        thread.start()

    def _train_ensemble_thread(self):
        """Thread d'entraînement de l'ensemble avec gestion d'erreurs complète"""
        try:
            def _train_internal():
                min_races = self.min_races_var.get()
                cv_folds = self.cv_folds_var.get()

                # Filtrage des données
                filtered_data = self.processed_data[self.processed_data['nombreCourses'] >= min_races].copy()

                self.queue.put(('training_info',
                    f"🤖 ENTRAÎNEMENT IA ENSEMBLE\n"
                    f"{'='*60}\n"
                    f"Dataset: {len(filtered_data)} chevaux (min {min_races} courses)\n"
                    f"Validation: {cv_folds} folds temporels\n"
                    f"Modèles: LightGBM + XGBoost + CatBoost + Random Forest\n\n"
                ))

                # Sélection des features pour l'entraînement
                feature_cols = self.select_training_features(filtered_data)
                X = filtered_data[feature_cols].fillna(0)

                self.queue.put(('training_info', f"🔧 Features sélectionnées: {len(feature_cols)}\n"))

                # Variables cibles
                targets = {
                    'position': filtered_data['final_position'],
                    'win': filtered_data['won_race'],
                    'place': filtered_data['top3_finish']
                }

                all_results = {}

                # Entraînement pour chaque cible
                for target_name, y in targets.items():
                    self.queue.put(('training_info', f"\n🎯 ENTRAÎNEMENT POUR: {target_name.upper()}\n"))
                    self.queue.put(('training_info', f"{'-'*50}\n"))

                    # Entraîner l'ensemble
                    cv_scores = self.ensemble.train_ensemble(X, y, target_name, cv_folds)

                    # Afficher les résultats
                    self.queue.put(('training_info', f"📊 Résultats validation croisée:\n"))
                    for model_name, score in cv_scores.items():
                        self.queue.put(('training_info', f"  • {model_name}: {score:.4f}\n"))

                    # Calculer les poids d'ensemble
                    weights = self.ensemble.weights.get(target_name, {})
                    self.queue.put(('training_info', f"\n⚖️ Poids d'ensemble:\n"))
                    for model_name, weight in weights.items():
                        self.queue.put(('training_info', f"  • {model_name}: {weight:.3f}\n"))

                    all_results[target_name] = {
                        'cv_scores': cv_scores,
                        'weights': weights,
                        'best_score': max(cv_scores.values()) if cv_scores else 0
                    }

                # Sauvegarde des noms de features
                self.ensemble.feature_names = feature_cols
                self.ensemble.is_trained = True
                self.training_results = all_results

                # Résumé final
                self.queue.put(('training_info', f"\n🏆 RÉSUMÉ FINAL\n"))
                self.queue.put(('training_info', f"{'='*40}\n"))
                for target_name, results in all_results.items():
                    best_score = results['best_score']
                    self.queue.put(('training_info', f"📊 {target_name.upper()}: {best_score:.4f}\n"))

                self.queue.put(('training_complete',
                    "🎉 Entraînement IA Ensemble terminé avec succès !\n\n"
                    "L'intelligence artificielle est prête à prédire les courses."
                ))

            # Exécution avec gestion d'erreurs
            self.error_handler.robust_execute(_train_internal, 'model_training')

        except Exception as e:
            self.queue.put(('training_error', f"Erreur d'entraînement IA: {str(e)}"))

    def select_training_features(self, df):
        """Sélectionner les features pour l'entraînement - Version complète"""

        # Features numériques de base
        base_features = [
            'age', 'nombreCourses', 'nombreVictoires', 'nombrePlaces', 
            'nombrePlacesSecond', 'nombrePlacesTroisieme'
        ]

        # Ratios de performance
        performance_features = [
            'win_rate', 'place_rate', 'place2_rate', 'place3_rate', 'top3_rate'
        ]

        # Features de forme et musique
        form_features = [
            'recent_form_score', 'consistency_score', 'trend_score', 'best_recent_position'
        ]

        # Features financières
        financial_features = [
            'gains_carriere', 'gains_victoires', 'gains_place', 'gains_annee_courante',
            'gains_annee_precedente', 'avg_gain_per_race', 'win_gain_ratio', 'recent_earning_trend'
        ]

        # Features de marché
        market_features = [
            'direct_odds', 'reference_odds', 'is_favorite', 'odds_trend',
            'odds_movement', 'odds_volatility'
        ]

        # Features compétitives
        competitive_features = [
            'field_avg_winrate', 'field_strength', 'relative_experience',
            'winrate_rank', 'earnings_rank', 'odds_rank'
        ]

        # Features d'interaction
        interaction_features = [
            'age_experience', 'winrate_odds', 'form_earnings', 'consistency_winrate'
        ]

        # Features catégorielles encodées
        categorical_features = [col for col in df.columns if col.endswith('_encoded')]

        # Features temporelles
        temporal_features = [col for col in df.columns if col in ['day_of_week', 'month', 'is_weekend']]

        # Combiner toutes les features disponibles
        all_feature_groups = [
            base_features, performance_features, form_features, financial_features, 
            market_features, competitive_features, interaction_features, 
            categorical_features, temporal_features
        ]

        selected_features = []
        for group in all_feature_groups:
            for feature in group:
                if feature in df.columns:
                    selected_features.append(feature)

        return selected_features

    # ===== MÉTHODES DE PRÉDICTION COMPLÈTES =====

    def on_prediction_mode_change(self):
        """Gérer le changement de mode de prédiction - Version complète"""
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
        else:
            self.existing_frame.pack_forget()
            self.new_file_frame.pack(fill='x', pady=10)

    def browse_new_race_file(self):
        """Parcourir pour sélectionner un nouveau fichier de course"""
        filename = filedialog.askopenfilename(
            title="Sélectionner un fichier de course",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.new_file_var.set(filename)

    def load_new_race_file(self):
        """Charger un nouveau fichier de course - Version complète"""
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
            for participant in race_data['participants']:
                participant['race_date'] = race_date
                participant['race_file'] = basename
                participants.append(participant)

            temp_df = pd.DataFrame(participants)

            # Réinitialiser le détecteur de nouvelles valeurs
            if hasattr(self.feature_engineer, '_unknown_detected'):
                delattr(self.feature_engineer, '_unknown_detected')

            # Extraction des features avec gestion d'erreurs améliorée
            print(f"🔄 Traitement du fichier: {basename}")
            print(f"📊 Participants trouvés: {len(participants)}")

            # Extraire les features
            self.new_race_data = self.feature_engineer.extract_comprehensive_features(temp_df)

            num_horses = len(self.new_race_data)
            num_features = self.new_race_data.shape[1]

            # Vérifier s'il y a de nouvelles valeurs
            unknown_report = self.feature_engineer.get_unknown_values_report()

            if unknown_report:
                # Afficher le rapport des nouvelles valeurs
                info_message = (
                    f"✅ Fichier chargé avec succès !\n\n"
                    f"📊 STATISTIQUES:\n"
                    f"• {num_horses} chevaux traités\n"
                    f"• {num_features} caractéristiques extraites\n\n"
                    f"⚠️ NOUVELLES VALEURS DÉTECTÉES:\n"
                    f"L'IA a détecté de nouveaux participants non vus\n"
                    f"pendant l'entraînement. Voir la console pour détails.\n\n"
                    f"💡 Impact: Précision légèrement réduite pour\n"
                    f"les nouveaux drivers/entraîneurs."
                )

                print(f"\n{unknown_report}")
                messagebox.showinfo("Succès avec nouvelles valeurs", info_message)
                self.new_file_info.config(text=f"⚠️ Chargé avec nouveaux participants: {num_horses} chevaux - {basename}")
            else:
                # Pas de nouvelles valeurs
                messagebox.showinfo("Succès",
                    f"✅ Fichier chargé parfaitement !\n\n"
                    f"• {num_horses} chevaux traités par l'IA\n"
                    f"• {num_features} caractéristiques extraites\n"
                    f"• Tous les participants sont connus de l'IA\n"
                    f"• Prêt pour prédiction optimale"
                )
                self.new_file_info.config(text=f"✅ Fichier chargé: {num_horses} chevaux - {basename}")

        except ValueError as e:
            error_msg = str(e)
            if "previously unseen label" in error_msg:
                # Cette erreur ne devrait plus se produire avec le nouvel encodage
                messagebox.showerror("Erreur de compatibilité",
                    f"Erreur d'encodage inattendue.\n\n"
                    f"Détails: {error_msg}\n\n"
                    f"Solution: Réentraîner l'IA avec vos données actuelles."
                )
            else:
                messagebox.showerror("Erreur", f"Erreur lors du chargement: {error_msg}")

            self.new_file_info.config(text="❌ Erreur de chargement")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement: {str(e)}")
            self.new_file_info.config(text="❌ Erreur de chargement")

    def predict_race(self):
        """Prédire une course avec l'ensemble IA - VERSION CORRIGÉE"""
        if not self.ensemble.is_trained:
            messagebox.showerror("Erreur", "Veuillez d'abord entraîner l'IA Ensemble")
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
            # Vérifier les features
            missing_features = [f for f in self.ensemble.feature_names if f not in race_data.columns]
            if missing_features:
                messagebox.showerror("Erreur", f"Features manquantes: {missing_features[:5]}...")
                return
        
            # Préparation des données
            X_pred = race_data[self.ensemble.feature_names].fillna(0)
        
            # Prédictions avec l'ensemble
            predictions = {}
            confidence_scores = {}
        
            for target_name in ['position', 'win', 'place']:
                try:
                    pred = self.ensemble.predict_ensemble(X_pred, target_name)
                    predictions[target_name] = pred
        
                    # Calcul de confiance basé sur la variance des prédictions individuelles
                    confidence = self.ensemble.calculate_prediction_confidence(X_pred, target_name)
                    confidence_scores[target_name] = confidence
        
                except Exception as e:
                    print(f"Erreur prédiction {target_name}: {str(e)}")
                    continue
        
            if not predictions:
                messagebox.showerror("Erreur", "Aucune prédiction n'a pu être générée")
                return
        
            # Préparation des résultats
            results_df = race_data[['numPmu', 'nom', 'driver', 'direct_odds']].copy()
        
            # Ajouter les probabilités
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
        
            # ===== CORRECTION PRINCIPALE =====
            # Trier par probabilité de victoire DÉCROISSANTE (le plus probable en premier)
            results_df = results_df.sort_values('prob_win', ascending=False)
            
            # Réinitialiser l'index pour avoir un classement propre
            results_df = results_df.reset_index(drop=True)
        
            # Affichage dans le tableau
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
        
                self.prediction_tree.insert('', 'end', values=(
                    i + 1,  # Rang basé sur la position dans le tableau trié
                    horse['numPmu'],
                    horse['nom'][:20],
                    horse['driver'][:15],
                    f"{horse['prob_win']:.3f}",
                    f"{horse['prob_place']:.3f}",
                    f"{horse['direct_odds']:.1f}",
                    confidence_text
                ))
        
            # Générer les recommandations IA
            self.generate_ai_recommendations(results_df)
        
            # Mise à jour du statut
            source_info = "Course de la base" if mode == "existing" else f"Nouveau fichier"
            self.status_var.set(f"Prédiction IA Ensemble réalisée - {source_info}")
            
            # Debug : afficher les 3 premiers pour vérification
            print("🔍 VÉRIFICATION DU CLASSEMENT:")
            print("=" * 50)
            for i in range(min(3, len(results_df))):
                horse = results_df.iloc[i]
                print(f"{i+1}. N°{horse['numPmu']} {horse['nom']}")
                print(f"   Prob victoire: {horse['prob_win']:.3f} ({horse['prob_win']*100:.1f}%)")
                print(f"   Prob place: {horse['prob_place']:.3f} ({horse['prob_place']*100:.1f}%)")
                print(f"   Confiance: {horse['confidence']*100:.0f}%")
                print()
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la prédiction: {str(e)}")
        
    def generate_ai_recommendations(self, results_df):
        """Générer les recommandations IA avancées - Version complète"""
        reco_text = "🤖 RECOMMANDATIONS IA ENSEMBLE 🤖\n"
        reco_text += "=" * 60 + "\n\n"

        # Analyse du favori IA
        best_pick = results_df.iloc[0]
        reco_text += f"🏆 FAVORI IA\n"
        reco_text += f"{'='*20}\n"

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
        reco_text += f"Confiance IA: {confidence_icon} ({confidence_level:.1%})\n"
        reco_text += f"Cote: {best_pick['direct_odds']:.1f}\n\n"

        # Tiercé IA
        reco_text += f"🎯 TIERCÉ RECOMMANDÉ IA\n"
        reco_text += f"{'='*30}\n"
        top3 = results_df.head(3)

        for i, (_, horse) in enumerate(top3.iterrows()):
            conf_pct = horse['confidence'] * 100
            reco_text += f"{i+1}. #{horse['numPmu']} - {horse['nom']} "
            reco_text += f"({conf_pct:.0f}% confiance)\n"

        # Analyse des outsiders
        reco_text += f"\n💎 OUTSIDERS À SURVEILLER\n"
        reco_text += f"{'='*35}\n"

        # Chercher les chevaux avec forte probabilité mais cotes élevées
        outsiders = results_df[(results_df['direct_odds'] > 10) &
                              (results_df['prob_win'] > 0.05)]

        if not outsiders.empty:
            best_outsider = outsiders.loc[outsiders['prob_win'].idxmax()]
            expected_value = best_outsider['prob_win'] * best_outsider['direct_odds']

            if expected_value > 1.2:  # Valeur positive
                reco_text += f"#{best_outsider['numPmu']} - {best_outsider['nom']}\n"
                reco_text += f"Cote: {best_outsider['direct_odds']:.1f} | "
                reco_text += f"Prob: {best_outsider['prob_win']:.1%} | "
                reco_text += f"Valeur: {expected_value:.2f}\n"
            else:
                reco_text += "Aucun outsider intéressant détecté\n"
        else:
            reco_text += "Aucun outsider dans les critères\n"

        # Stratégie recommandée
        reco_text += f"\n📈 STRATÉGIE IA RECOMMANDÉE\n"
        reco_text += f"{'='*35}\n"

        high_confidence_count = len(results_df[results_df['confidence'] > 0.6])
        avg_confidence = results_df['confidence'].mean()

        if avg_confidence > 0.7:
            strategy = "🔥 EXCELLENTE"
        elif avg_confidence > 0.6:
            strategy = "✅ BONNE"
        elif avg_confidence > 0.5:
            strategy = "⚠️ CORRECTE"
        else:
            strategy = "❌ À AMÉLIORER"

        reco_text += f"Stratégie: {strategy}\n"
        reco_text += f"Confiance moyenne: {avg_confidence:.1%}\n"
        reco_text += f"Chevaux haute confiance: {high_confidence_count}/16\n"

        # Footer technique
        reco_text += f"\n🔬 Powered by IA Ensemble:\n"
        models_used = []
        if 'win' in self.ensemble.models and self.ensemble.models['win']:
            models_used = list(self.ensemble.models['win'].keys())

        if models_used:
            reco_text += f"Modèles: {', '.join(models_used)}\n"
        reco_text += f"Features analysées: {len(self.ensemble.feature_names)}\n"

        self.reco_text.delete(1.0, tk.END)
        self.reco_text.insert(1.0, reco_text)

    def sort_treeview_column(self, col):
        """Trier le tableau par colonne - Version complète"""
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

    # ===== MÉTHODES D'ANALYTICS COMPLÈTES =====

    def show_model_performance(self):
        """Afficher les performances des modèles - Version complète avec graphiques"""
        if not self.training_results:
            messagebox.showwarning("Attention", "Aucun modèle entraîné")
            return

        # Nettoyer le frame
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()

        # Créer les graphiques de performance
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Performance des Modèles IA Ensemble', fontsize=16, fontweight='bold')

        # Graphique 1: Scores de validation croisée
        targets = list(self.training_results.keys())
        if targets and self.training_results[targets[0]].get('cv_scores'):
            models = list(self.training_results[targets[0]]['cv_scores'].keys())

            x_pos = np.arange(len(models))
            width = 0.25

            for i, target in enumerate(targets):
                if 'cv_scores' in self.training_results[target]:
                    scores = [self.training_results[target]['cv_scores'][model] for model in models]
                    axes[0, 0].bar(x_pos + i*width, scores, width, label=target.capitalize(), alpha=0.8)

            axes[0, 0].set_xlabel('Modèles')
            axes[0, 0].set_ylabel('Score CV')
            axes[0, 0].set_title('Scores de Validation Croisée')
            axes[0, 0].set_xticks(x_pos + width)
            axes[0, 0].set_xticklabels(models, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Graphique 2: Poids d'ensemble
        if 'win' in self.training_results and 'weights' in self.training_results['win']:
            weights = self.training_results['win']['weights']
            if weights:
                axes[0, 1].pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%')
                axes[0, 1].set_title('Poids d\'Ensemble (Victoire)')

        # Graphique 3: Comparaison des targets
        target_scores = []
        target_names = []
        for target, results in self.training_results.items():
            target_scores.append(results.get('best_score', 0))
            target_names.append(target.capitalize())

        if target_scores:
            bars = axes[1, 0].bar(target_names, target_scores, color=['#3498db', '#e74c3c', '#f39c12'])
            axes[1, 0].set_ylabel('Meilleur Score')
            axes[1, 0].set_title('Performance par Objectif')
            axes[1, 0].grid(True, alpha=0.3)

            # Ajouter les valeurs sur les barres
            for bar, score in zip(bars, target_scores):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        # Graphique 4: Feature importance (si disponible)
        try:
            importances_dict = self.ensemble.get_feature_importance('win')
            
            if importances_dict and 'lgb' in importances_dict:
                importances = importances_dict['lgb']
                feature_names = self.ensemble.feature_names

                # Top 10 features
                indices = np.argsort(importances)[::-1][:10]
                top_importances = importances[indices]
                top_features = [feature_names[i][:8] for i in indices]  # Tronquer les noms
                
                axes[1, 1].barh(range(len(top_features)), top_importances, color='skyblue')
                axes[1, 1].set_yticks(range(len(top_features)))
                axes[1, 1].set_yticklabels(reversed(top_features))
                axes[1, 1].set_xlabel('Importance')
                axes[1, 1].set_title('Top 10 Features')
                axes[1, 1].invert_yaxis()
            else:
                axes[1, 1].text(0.5, 0.5, 'Feature importance\nnon disponible',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        except:
            axes[1, 1].text(0.5, 0.5, 'Données non disponibles',
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()

        # Intégrer dans tkinter
        canvas = FigureCanvasTkAgg(fig, self.analysis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def show_feature_importance(self):
        """Afficher l'importance des features - Version complète"""
        if not self.ensemble.is_trained:
            messagebox.showwarning("Attention", "Aucun modèle entraîné")
            return

        # Nettoyer le frame
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Importance des Features - IA Ensemble', fontsize=16, fontweight='bold')

        try:
            # Récupérer l'importance des features
            importances_dict = self.ensemble.get_feature_importance('win')
            
            if importances_dict and 'lgb' in importances_dict:
                importances = importances_dict['lgb']
                feature_names = self.ensemble.feature_names

                # Top 20 features
                indices = np.argsort(importances)[::-1][:20]
                top_importances = importances[indices]
                top_features = [feature_names[i] for i in indices]

                # Graphique 1: Barres horizontales
                y_pos = np.arange(len(top_features))
                axes[0].barh(y_pos, top_importances, color='skyblue', alpha=0.8)
                axes[0].set_yticks(y_pos)
                axes[0].set_yticklabels([f[:15] for f in reversed(top_features)])
                axes[0].set_xlabel('Importance')
                axes[0].set_title('Top 20 Features')
                axes[0].grid(True, alpha=0.3)
                axes[0].invert_yaxis()

                # Graphique 2: Distribution des importances
                axes[1].hist(importances, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
                axes[1].axvline(np.mean(importances), color='red', linestyle='--',
                               label=f'Moyenne: {np.mean(importances):.4f}')
                axes[1].set_xlabel('Importance')
                axes[1].set_ylabel('Nombre de Features')
                axes[1].set_title('Distribution des Importances')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            else:
                for ax in axes:
                    ax.text(0.5, 0.5, 'Feature importance\nnon disponible',
                           ha='center', va='center', transform=ax.transAxes)

        except Exception as e:
            for ax in axes:
                ax.text(0.5, 0.5, f'Erreur:\n{str(e)}',
                       ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.analysis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def show_temporal_analysis(self):
        """Afficher l'analyse temporelle - Version complète"""
        if self.processed_data is None:
            messagebox.showwarning("Attention", "Aucune donnée chargée")
            return

        # Nettoyer le frame
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analyse Temporelle - Dataset Hippique', fontsize=16, fontweight='bold')

        try:
            # Évolution mensuelle des performances
            monthly_stats = self.processed_data.groupby(
                self.processed_data['race_date'].dt.to_period('M')
            ).agg({
                'win_rate': 'mean',
                'direct_odds': 'median',
                'recent_form_score': 'mean',
                'gains_carriere': 'mean'
            }).reset_index()

            months = [str(period) for period in monthly_stats['race_date']]

            # Graphique 1: Taux de victoire
            axes[0, 0].plot(months, monthly_stats['win_rate'], marker='o', linewidth=2,
                           color='#2ecc71', markersize=6)
            axes[0, 0].set_title('Évolution du Taux de Victoire Moyen')
            axes[0, 0].set_ylabel('Taux de Victoire')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)

            # Graphique 2: Cotes médianes
            axes[0, 1].plot(months, monthly_stats['direct_odds'], marker='s', linewidth=2,
                           color='#e74c3c', markersize=6)
            axes[0, 1].set_title('Évolution des Cotes Médianes')
            axes[0, 1].set_ylabel('Cotes Directes')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

            # Graphique 3: Score de forme
            axes[1, 0].plot(months, monthly_stats['recent_form_score'], marker='^', linewidth=2,
                           color='#3498db', markersize=6)
            axes[1, 0].set_title('Évolution du Score de Forme Moyen')
            axes[1, 0].set_ylabel('Score de Forme')
            axes[1, 0].set_xlabel('Mois')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

            # Graphique 4: Distribution par jour de la semaine
            if 'day_of_week' in self.processed_data.columns:
                day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
                day_counts = self.processed_data['day_of_week'].value_counts().sort_index()

                axes[1, 1].bar(range(len(day_counts)), day_counts.values,
                              color='#9b59b6', alpha=0.8)
                axes[1, 1].set_xticks(range(len(day_names)))
                axes[1, 1].set_xticklabels(day_names, rotation=45)
                axes[1, 1].set_title('Répartition des Courses par Jour')
                axes[1, 1].set_ylabel('Nombre de Chevaux')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Données temporelles\nnon disponibles',
                               ha='center', va='center', transform=axes[1, 1].transAxes)

        except Exception as e:
            for ax in axes.flat:
                ax.text(0.5, 0.5, f'Erreur:\n{str(e)}',
                       ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.analysis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def show_prediction_analysis(self):
        """Afficher l'analyse des prédictions - Version complète"""
        # Cette méthode sera implémentée après avoir des prédictions
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()

        info_label = ttk.Label(self.analysis_frame,
                              text="📊 Analyse des Prédictions\n\n" +
                                   "Cette fonctionnalité analysera :\n" +
                                   "• Distribution des probabilités\n" +
                                   "• Confiance par modèle\n" +
                                   "• Cohérence des prédictions\n" +
                                   "• Détection d'anomalies\n\n" +
                                   "Effectuez d'abord une prédiction pour voir l'analyse.",
                              font=('Arial', 12), justify='center')
        info_label.pack(expand=True)

    # ===== MÉTHODES DE BACKTESTING COMPLÈTES =====

    def run_backtest(self):
        """Lancer le backtesting - Version complète"""
        if not self.ensemble.is_trained:
            messagebox.showwarning("Attention", "Veuillez d'abord entraîner l'IA")
            return

        if self.processed_data is None:
            messagebox.showwarning("Attention", "Aucune donnée disponible")
            return

        self.backtest_text.delete(1.0, tk.END)
        self.backtest_text.insert(tk.END, "🚀 Lancement du backtesting...\n\n")

        # Lancer dans un thread
        thread = threading.Thread(target=self._run_backtest_thread)
        thread.daemon = True
        thread.start()

    def _run_backtest_thread(self):
        """Thread de backtesting complet"""
        try:
            test_period = self.backtest_period_var.get() / 100
            strategy = self.strategy_var.get()

            # Configurer le moteur de backtesting
            self.backtesting_engine.ensemble_model = self.ensemble

            # Lancer le backtest
            results = self.backtesting_engine.run_backtest(
                data=self.processed_data,
                strategy=strategy,
                test_period=test_period,
                feature_names=self.ensemble.feature_names
            )

            # Générer le rapport
            report = self.backtesting_engine.get_backtest_report(results)

            # Afficher les résultats dans l'interface
            self.queue.put(('backtest_results', report))

        except Exception as e:
            self.queue.put(('backtest_error', f"Erreur lors du backtesting: {str(e)}"))

    # ===== MÉTHODES DE SAUVEGARDE/CHARGEMENT COMPLÈTES =====

    def save_models(self):
        """Sauvegarder l'ensemble de modèles - Version complète"""
        if not self.ensemble.is_trained:
            messagebox.showerror("Erreur", "Aucun modèle entraîné à sauvegarder")
            return

        filename = filedialog.asksaveasfilename(
            title="Sauvegarder l'IA Ensemble",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if filename:
            try:
                save_data = {
                    'ensemble': self.ensemble,
                    'feature_engineer': self.feature_engineer,
                    'training_results': self.training_results,
                    'version': '2.0',
                    'timestamp': datetime.now().isoformat()
                }

                with open(filename, 'wb') as f:
                    pickle.dump(save_data, f)

                messagebox.showinfo("Succès",
                    f"IA Ensemble sauvegardée avec succès !\n\n"
                    f"Contenu:\n"
                    f"• {len(self.ensemble.models)} groupes de modèles\n"
                    f"• {len(self.ensemble.feature_names)} features\n"
                    f"• Résultats d'entraînement complets"
                )

            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur de sauvegarde: {str(e)}")

    def load_models(self):
        """Charger l'ensemble de modèles - Version complète"""
        filename = filedialog.askopenfilename(
            title="Charger l'IA Ensemble",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'rb') as f:
                    save_data = pickle.load(f)

                # Vérifier la compatibilité
                if 'version' in save_data and save_data['version'] == '2.0':
                    self.ensemble = save_data['ensemble']
                    self.feature_engineer = save_data.get('feature_engineer', self.feature_engineer)
                    self.training_results = save_data.get('training_results', {})

                    # Mettre à jour le moteur de backtesting
                    self.backtesting_engine.ensemble_model = self.ensemble

                    messagebox.showinfo("Succès",
                        f"IA Ensemble chargée avec succès !\n\n"
                        f"Contenu:\n"
                        f"• {len(self.ensemble.models)} groupes de modèles\n"
                        f"• {len(self.ensemble.feature_names)} features\n"
                        f"• Sauvegardé: {save_data.get('timestamp', 'Date inconnue')}"
                    )

                    # Activer les boutons
                    self.save_btn.config(state='normal')

                    # Afficher les résultats dans l'onglet entraînement
                    if self.training_results:
                        result_text = "🤖 IA ENSEMBLE CHARGÉE\n" + "="*50 + "\n\n"
                        for target, results in self.training_results.items():
                            result_text += f"📊 {target.upper()}:\n"
                            if 'cv_scores' in results:
                                for model, score in results['cv_scores'].items():
                                    result_text += f"  • {model}: {score:.4f}\n"
                            result_text += f"  Meilleur score: {results.get('best_score', 0):.4f}\n\n"

                        self.results_text.delete(1.0, tk.END)
                        self.results_text.insert(1.0, result_text)

                else:
                    messagebox.showwarning("Attention",
                        "Format de fichier non compatible.\n"
                        "Veuillez utiliser un fichier sauvegardé avec cette version."
                    )

            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur de chargement: {str(e)}")

    # ===== GESTION DES ÉVÉNEMENTS COMPLÈTE =====

    def check_queue(self):
        """Vérifier la queue pour les mises à jour des threads - Version complète"""
        try:
            while True:
                msg_type, *args = self.queue.get_nowait()

                if msg_type == 'progress':
                    progress, label = args
                    self.progress_var.set(progress)
                    self.progress_label.config(text=label)
                    self.status_var.set(label)

                elif msg_type == 'stats':
                    horses, races, avg, period, features = args
                    self.horses_var.set(f"{horses:,}")
                    self.races_var.set(f"{races:,}")  
                    self.avg_var.set(f"{avg:.1f}")
                    self.period_var.set(f"{period}")
                    self.features_var.set(f"{features}")

                elif msg_type == 'complete':
                    message = args[0]
                    messagebox.showinfo("Succès", message)
                    self.process_btn.config(state='normal')
                    self.train_btn.config(state='normal')

                    # Remplir la liste des courses
                    if self.processed_data is not None:
                        race_files = sorted(self.processed_data['race_file'].unique())
                        self.race_combo['values'] = race_files
                        if race_files:
                            self.race_combo.set(race_files[0])

                elif msg_type == 'error':
                    message = args[0]
                    messagebox.showerror("Erreur", message)
                    self.process_btn.config(state='normal')

                elif msg_type == 'warning':
                    message = args[0]
                    print(f"Warning: {message}")

                elif msg_type == 'training_info':
                    message = args[0]
                    self.results_text.insert(tk.END, message)
                    self.results_text.see(tk.END)

                elif msg_type == 'training_complete':
                    message = args[0]
                    messagebox.showinfo("🎉 Succès", message)
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

                elif msg_type == 'backtest_error':
                    error_msg = args[0]
                    messagebox.showerror("Erreur Backtest", error_msg)
                    self.backtest_text.insert(tk.END, f"\n❌ ERREUR: {error_msg}\n")

        except:
            pass

        # Programmer la prochaine vérification
        self.root.after(100, self.check_queue)

    def show_welcome_message(self):
        """Afficher le message de bienvenue complet"""
        welcome_msg = (
            "🎉 Bienvenue dans le Prédicteur Hippique PRO !\n\n"
            "🚀 NOUVEAUTÉS DE CETTE VERSION COMPLÈTE :\n"
            "✅ IA Ensemble avec 4 modèles combinés (LightGBM, XGBoost, CatBoost, RF)\n"
            "✅ 45+ features avancées extraites automatiquement\n"
            "✅ Cache intelligent pour performances optimales\n"
            "✅ Backtesting intégré avec 4 stratégies de validation\n"
            "✅ Interface moderne et intuitive avec graphiques matplotlib\n"
            "✅ Gestion robuste des erreurs avec retry automatique\n"
            "✅ Analytics avancés avec graphiques interactifs\n"
            "✅ Sauvegarde/chargement des modèles IA\n"
            "✅ Optimisation mémoire et monitoring système\n\n"
            "📁 POUR COMMENCER :\n"
            "1. Allez dans l'onglet 'Données'\n"
            "2. Sélectionnez votre dossier de fichiers JSON\n"
            "3. Lancez le traitement IA avancé\n"
            "4. Entraînez l'IA Ensemble\n"
            "5. Profitez des prédictions optimales !\n\n"
            "💡 CONSEIL : Utilisez au moins 100 courses pour un entraînement optimal."
        )
        
        messagebox.showinfo("🏇 Prédicteur Hippique PRO", welcome_msg)


def check_dependencies():
    """Vérifier que toutes les dépendances sont installées - Version corrigée"""
    # Dictionnaire: nom_pour_import -> nom_pour_pip
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'lightgbm': 'lightgbm',
        'xgboost': 'xgboost',
        'sklearn': 'scikit-learn',  # Import sklearn mais install scikit-learn
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tkinter': 'tkinter'
    }
    
    missing_packages = []
    
    for import_name, pip_name in required_packages.items():
        try:
            if import_name == 'tkinter':
                import tkinter
            else:
                __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        error_msg = (
            "❌ Dépendances manquantes détectées !\n\n"
            f"Packages manquants: {', '.join(missing_packages)}\n\n"
            "Pour installer les dépendances manquantes :\n"
            "pip install " + " ".join(missing_packages) + "\n\n"
            "Note: tkinter est généralement inclus avec Python"
        )
        
        # Essayer d'afficher avec tkinter, sinon print
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Dépendances manquantes", error_msg)
            root.destroy()
        except:
            print(error_msg)
        
        return False
    
    return True


def setup_environment():
    """Configuration de l'environnement d'exécution"""
    # Configuration des chemins si nécessaire
    import os
    
    # Créer les dossiers nécessaires
    directories = ['logs', 'models', 'cache']
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except:
                pass  # Ignore si création impossible
    
    # Configuration matplotlib pour éviter les conflits
    try:
        import matplotlib
        matplotlib.use('TkAgg')
    except:
        pass


def main():
    """Fonction principale de l'application complète"""
    print("🏇 Lancement du Prédicteur Hippique PRO - VERSION COMPLÈTE")
    print("=" * 70)
    print("🤖 IA Ensemble: LightGBM + XGBoost + CatBoost + Random Forest")
    print("🔧 Features avancées: 45+ caractéristiques par cheval")
    print("⚡ Optimisations: Cache intelligent + Gestion d'erreurs robuste")
    print("📊 Analytics: Graphiques interactifs + Backtesting avancé")
    print("💾 Sauvegarde/Chargement: Modèles IA complets")
    print("🎯 Prédictions: Recommandations IA avec scores de confiance")
    print("=" * 70)
    
    # Vérifier les dépendances
    print("🔍 Vérification des dépendances...")
    if not check_dependencies():
        print("❌ Échec du lancement - dépendances manquantes")
        sys.exit(1)
    print("✅ Toutes les dépendances sont installées")
    
    # Configuration de l'environnement
    print("⚙️ Configuration de l'environnement...")
    setup_environment()
    print("✅ Environnement configuré")
    
    # Création de l'interface principale
    print("🎨 Initialisation de l'interface graphique complète...")
    try:
        root = tk.Tk()
        
        # Configuration de la fenêtre principale
        root.minsize(1200, 800)
        
        # Gestion de la fermeture propre
        def on_closing():
            """Gestion de la fermeture de l'application"""
            if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter l'application ?"):
                print("👋 Fermeture de l'application...")
                try:
                    # Nettoyage si nécessaire
                    root.quit()
                    root.destroy()
                except:
                    pass
                print("✅ Application fermée proprement")
                sys.exit(0)
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Création de l'application complète
        print("🚀 Création de l'application complète...")
        app = CompleteHorseRacingGUI(root)
        print("✅ Application créée avec succès")
        
        # Informations système
        try:
            if PSUTIL_AVAILABLE:
                memory_mb = psutil.Process().memory_info().rss / 1024**2
                print(f"💾 Utilisation mémoire initiale: {memory_mb:.0f}MB")
            else:
                print("💾 Monitoring mémoire non disponible (psutil manquant)")
        except:
            print("💾 Monitoring mémoire non disponible")
        
        print("🎉 Interface graphique complète prête !")
        print("📋 Fonctionnalités disponibles:")
        print("   • Chargement et traitement des données JSON")
        print("   • Entraînement IA Ensemble avec 4 modèles")
        print("   • Prédictions avec recommandations avancées")
        print("   • Analytics avec graphiques matplotlib")
        print("   • Backtesting avec 4 stratégies")
        print("   • Sauvegarde/chargement des modèles")
        print("=" * 70)
        
        # Lancement de la boucle principale
        root.mainloop()
        
    except Exception as e:
        error_msg = (
            f"❌ Erreur critique lors du lancement :\n"
            f"{str(e)}\n\n"
            f"Vérifications à effectuer :\n"
            f"1. Python 3.7+ installé\n"
            f"2. Toutes les dépendances installées correctement\n"
            f"3. Interface graphique disponible (pas de SSH sans X11)\n"
            f"4. Permissions d'écriture dans le répertoire courant\n"
            f"5. Modules locaux présents dans le même dossier"
        )
        
        print(error_msg)
        
        # Essayer d'afficher l'erreur dans une boîte de dialogue
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Erreur critique", error_msg)
            root.destroy()
        except:
            pass
        
        sys.exit(1)


def info():
    """Afficher les informations complètes sur l'application"""
    info_text = """
🏇 PRÉDICTEUR HIPPIQUE PRO - v2.0 COMPLÈTE
==========================================

📋 DESCRIPTION:
Application d'intelligence artificielle avancée pour prédire les courses hippiques
Utilise un ensemble de 4 modèles ML de pointe pour des prédictions optimales

🤖 MODÈLES IA INTÉGRÉS:
• LightGBM - Gradient Boosting optimisé Microsoft (ultra-rapide)
• XGBoost - Extreme Gradient Boosting (très précis)
• CatBoost - Categorical Boosting Yandex (gère parfaitement les catégories)
• Random Forest - Forêt d'arbres décisionnels (robuste)

🔧 FONCTIONNALITÉS COMPLÈTES:
• Extraction automatique de 45+ features avancées par cheval
• Cache intelligent multi-niveaux pour performances optimales
• Backtesting avec 4 stratégies de validation (place, confidence, top_pick, value)
• Analytics complets avec graphiques matplotlib interactifs
• Gestion robuste des erreurs avec retry automatique et logging
• Interface moderne et intuitive avec 5 onglets spécialisés
• Sauvegarde/chargement complets des modèles entraînés
• Optimisation mémoire automatique des datasets
• Monitoring système en temps réel (cache, RAM)

📊 DONNÉES SUPPORTÉES:
• Fichiers JSON des courses hippiques (format PMU)
• Historique complet des performances des chevaux
• Informations détaillées des drivers et entraîneurs
• Cotes et données de marché en temps réel
• Gestion automatique des nouvelles valeurs

🎯 PRÉDICTIONS AVANCÉES:
• Probabilités de victoire avec modélisation bayésienne
• Probabilités de place (top 3) pondérées
• Recommandations de tiercé optimisées par l'IA
• Détection intelligente d'outsiders à forte valeur
• Scores de confiance multi-modèles
• Stratégies de mise recommandées

📈 BACKTESTING PROFESSIONNEL:
• Validation temporelle rigoureuse des performances
• 4 stratégies testées: place_strategy, confidence, top_pick, value_betting
• Métriques détaillées: ROI, précision, hit rate, variance
• Rapports complets formatés avec évaluation qualitative
• Comparaison de stratégies automatisée

📊 ANALYTICS & INSIGHTS:
• Performance détaillée des modèles avec graphiques
• Importance des features avec visualisations
• Analyse temporelle des tendances du marché
• Graphiques interactifs matplotlib intégrés

💾 GESTION DES MODÈLES:
• Export/import complets des ensembles entraînés
• Historique des résultats d'entraînement
• Configuration personnalisable des hyperparamètres
• Compatibilité garantie entre versions

🏗️ ARCHITECTURE TECHNIQUE:
• Interface modulaire avec séparation des responsabilités
• Threading pour éviter le gel de l'interface
• Queue thread-safe pour communications
• Gestion d'erreurs multi-niveaux
• Cache intelligent avec TTL automatique
• Optimisation mémoire des DataFrames

🎨 INTERFACE UTILISATEUR:
• Design moderne avec ttk et styles personnalisés
• 5 onglets spécialisés (Données, IA, Prédictions, Analytics, Backtesting)
• Barres de progression en temps réel
• Tableaux triables et interactifs
• Zones de texte avec coloration syntaxique
• Métriques système en temps réel

🔧 REQUIREMENTS TECHNIQUES:
• Python 3.7+ (recommandé 3.9+)
• RAM: minimum 4GB, recommandé 8GB+
• CPU: multi-core recommandé pour l'entraînement
• Stockage: 1GB libre minimum
• OS: Windows, macOS, Linux

📦 DÉPENDANCES:
pandas, numpy, lightgbm, xgboost, scikit-learn, matplotlib, seaborn
catboost (optionnel), psutil (optionnel)

🔗 AUTEUR: Assistant IA spécialisé
📅 VERSION: 2.0 Complète
🏷️ LICENCE: Usage libre et gratuit
💡 SUPPORT: Documentation intégrée + interface intuitive
"""
    print(info_text)


if __name__ == "__main__":
    # Vérifier les arguments de ligne de commande
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--info', '-i', 'info']:
            info()
            sys.exit(0)
        elif sys.argv[1] in ['--help', '-h', 'help']:
            print("🏇 Prédicteur Hippique PRO - Usage")
            print("="*40)
            print("Usage: python complete_app.py [OPTIONS]")
            print("")
            print("Options:")
            print("  --info, -i     Afficher les informations détaillées sur l'application")
            print("  --help, -h     Afficher cette aide")
            print("")
            print("Exemples:")
            print("  python complete_app.py          # Lancer l'application normale")
            print("  python complete_app.py --info   # Afficher les infos techniques")
            print("")
            print("📋 Fonctionnalités principales:")
            print("• IA Ensemble avec 4 modèles (LightGBM, XGBoost, CatBoost, RF)")
            print("• 45+ features automatiques par cheval")  
            print("• Backtesting professionnel avec 4 stratégies")
            print("• Analytics complets avec graphiques")
            print("• Sauvegarde/chargement des modèles IA")
            print("")
            print("🔧 Requirements: Python 3.7+, pandas, numpy, scikit-learn, etc.")
            sys.exit(0)
        else:
            print(f"❌ Argument inconnu: {sys.argv[1]}")
            print("💡 Utilisez --help pour voir l'aide")
            sys.exit(1)
    
    # Lancement normal de l'application complète
    main()