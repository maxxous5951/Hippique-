"""
Interface graphique principale pour le prédicteur hippique
Interface moderne avec onglets pour chargement, entraînement, prédictions et analytics
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
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Imports des modules locaux
from cache_manager import IntelligentCache
from error_handler import RobustErrorHandler
from feature_engineer import AdvancedFeatureEngineer
from ensemble_models import HorseRacingEnsemble
from backtesting import BacktestingEngine

# Gestion optionnelle de psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil non disponible - monitoring système désactivé")


class OptimizedHorseRacingGUI:
    """Interface graphique optimisée pour le prédicteur hippique"""
    
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

        # Queue pour les mises à jour thread-safe
        self.queue = queue.Queue()

        # Variables d'interface
        self.new_race_data = None

        # Configuration du style
        self._setup_style()
        
        # Configuration de l'interface
        self.setup_ui()
        self.check_queue()

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
        """Configuration de l'interface utilisateur complète"""

        # Header principal
        self._create_header()

        # Notebook pour les onglets
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Création des onglets
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

        subtitle_label = ttk.Label(header_frame, text="IA Ensemble avec LightGBM + XGBoost + CatBoost",
                                  font=('Arial', 10), foreground='#7f8c8d')
        subtitle_label.pack(side='left', padx=(20, 0))

    def _create_status_bar(self):
        """Créer la barre de statut avec métriques"""
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
        """Créer l'onglet de chargement des données"""
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

        # Titre
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x', padx=20, pady=20)

        title = ttk.Label(title_frame, text="📁 Chargement des Données",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack()

        subtitle = ttk.Label(title_frame, text="Traitement avancé des fichiers JSON avec extraction de features IA",
                           font=('Arial', 10), foreground='#7f8c8d')
        subtitle.pack(pady=(5, 0))

        # Sélection du dossier
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

        # Zone d'informations
        info_frame = ttk.LabelFrame(main_frame, text="ℹ️ Informations détaillées", padding=15)
        info_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.info_text = scrolledtext.ScrolledText(info_frame, height=10, width=80,
                                                  font=('Consolas', 9))
        self.info_text.pack(fill='both', expand=True)

        # Barre de progression
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill='x', padx=20, pady=10)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100, length=400)
        self.progress_bar.pack(fill='x')

        self.progress_label = ttk.Label(progress_frame, text="", font=('Arial', 9))
        self.progress_label.pack(pady=(5, 0))

        # Bouton principal
        self.process_btn = ttk.Button(main_frame, text="🚀 Traiter avec IA Avancée",
                                     command=self.process_files,
                                     style='Accent.TButton', state='disabled')
        self.process_btn.pack(pady=20)

        # Statistiques
        self._create_stats_section(main_frame)

    def _create_stats_section(self, parent):
        """Créer la section des statistiques"""
        stats_frame = ttk.LabelFrame(parent, text="📊 Statistiques du dataset", padding=15)
        stats_frame.pack(fill='x', padx=20, pady=10)

        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack()

        # Variables des métriques
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
        """Créer l'onglet d'entraînement IA"""
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

        # Header
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill='x', padx=20, pady=20)

        title = ttk.Label(title_frame, text="🤖 Intelligence Artificielle Ensemble",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack()

        subtitle = ttk.Label(title_frame,
                           text="Combinaison optimale de LightGBM, XGBoost, CatBoost et Random Forest",
                           font=('Arial', 10), foreground='#7f8c8d')
        subtitle.pack(pady=(5, 0))

        # Configuration d'entraînement
        self._create_training_config(main_frame)

        # Bouton d'entraînement
        self.train_btn = ttk.Button(main_frame, text="🚀 Lancer l'Entraînement IA Ensemble",
                                   command=self.train_ensemble_models,
                                   style='Accent.TButton', state='disabled')
        self.train_btn.pack(pady=20)

        # Zone de résultats
        results_frame = ttk.LabelFrame(main_frame, text="📊 Résultats d'entraînement", padding=15)
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, font=('Consolas', 9))
        self.results_text.pack(fill='both', expand=True)

        # Boutons de sauvegarde
        self._create_model_buttons(main_frame)

    def _create_training_config(self, parent):
        """Créer la section de configuration d'entraînement"""
        config_frame = ttk.LabelFrame(parent, text="⚙️ Configuration d'entraînement", padding=15)
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

        ttk.Label(config_grid, text="Seuil de confiance:").grid(row=1, column=0, sticky='w', padx=5)
        self.confidence_var = tk.DoubleVar(value=0.6)
        confidence_spin = ttk.Spinbox(config_grid, from_=0.1, to=0.9, increment=0.1,
                                     textvariable=self.confidence_var, width=10)
        confidence_spin.grid(row=1, column=1, padx=5, sticky='w')

    def _create_model_buttons(self, parent):
        """Créer les boutons de gestion des modèles"""
        buttons_frame = ttk.Frame(parent)
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
        """Créer l'onglet de prédictions"""
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="🎯 Prédictions")

        main_frame = ttk.Frame(self.tab3)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Titre
        title = ttk.Label(main_frame, text="🎯 Prédictions IA Ensemble",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack(pady=(0, 20))

        # Mode de prédiction
        self._create_prediction_mode_section(main_frame)

        # Bouton de prédiction
        self.predict_btn = ttk.Button(main_frame, text="🎯 Prédire avec IA Ensemble",
                                     command=self.predict_race, style='Accent.TButton')
        self.predict_btn.pack(pady=10)

        # Résultats
        self._create_prediction_results_section(main_frame)

    def _create_prediction_mode_section(self, parent):
        """Créer la section de sélection du mode de prédiction"""
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

        ttk.Label(self.existing_frame, text="Course:").pack(side='left', padx=5)
        self.race_var = tk.StringVar()
        self.race_combo = ttk.Combobox(self.existing_frame, textvariable=self.race_var,
                                      state='readonly', width=50)
        self.race_combo.pack(side='left', fill='x', expand=True, padx=5)

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

        load_new_btn = ttk.Button(self.new_file_frame, text="📥 Charger le fichier",
                                 command=self.load_new_race_file)
        load_new_btn.pack(pady=5)

        self.new_file_info = ttk.Label(self.new_file_frame, text="", foreground='green')
        self.new_file_info.pack(pady=5)

        # Initialiser l'affichage
        self.on_prediction_mode_change()

    def _create_prediction_results_section(self, parent):
        """Créer la section des résultats de prédiction"""
        results_frame = ttk.LabelFrame(parent, text="Résultats IA Ensemble", padding=10)
        results_frame.pack(fill='both', expand=True, pady=10)

        # Tableau des résultats
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

        # Recommandations IA
        reco_frame = ttk.LabelFrame(parent, text="Recommandations IA", padding=10)
        reco_frame.pack(fill='x', pady=10)

        self.reco_text = scrolledtext.ScrolledText(reco_frame, height=8, font=('Consolas', 9))
        self.reco_text.pack(fill='both', expand=True)

    def _create_analytics_tab(self):
        """Créer l'onglet d'analyses"""
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="📊 Analytics")

        main_frame = ttk.Frame(self.tab4)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        title = ttk.Label(main_frame, text="📊 Analytics & Insights IA",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack(pady=(0, 20))

        # Boutons d'analyse
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill='x', pady=10)

        ttk.Button(buttons_frame, text="📈 Performance Modèles",
                  command=self.show_model_performance).pack(side='left', padx=5)

        ttk.Button(buttons_frame, text="🔗 Feature Importance",
                  command=self.show_feature_importance).pack(side='left', padx=5)

        ttk.Button(buttons_frame, text="📅 Évolution Temporelle",
                  command=self.show_temporal_analysis).pack(side='left', padx=5)

        # Zone d'analyse
        self.analysis_frame = ttk.Frame(main_frame)
        self.analysis_frame.pack(fill='both', expand=True, pady=10)

    def _create_backtesting_tab(self):
        """Créer l'onglet de backtesting"""
        self.tab5 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab5, text="📈 Backtesting")

        main_frame = ttk.Frame(self.tab5)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        title = ttk.Label(main_frame, text="📈 Backtesting & Validation",
                         font=('Arial', 18, 'bold'), foreground='#2c3e50')
        title.pack(pady=(0, 20))

        # Configuration du backtest
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

        # Bouton de backtest
        backtest_btn = ttk.Button(main_frame, text="🚀 Lancer le Backtest",
                                 command=self.run_backtest, style='Accent.TButton')
        backtest_btn.pack(pady=10)

        # Résultats du backtest
        backtest_results_frame = ttk.LabelFrame(main_frame, text="Résultats du backtest", padding=10)
        backtest_results_frame.pack(fill='both', expand=True, pady=10)

        self.backtest_text = scrolledtext.ScrolledText(backtest_results_frame, height=15,
                                                      font=('Consolas', 9))
        self.backtest_text.pack(fill='both', expand=True)

    # ===== MÉTHODES D'ÉVÉNEMENTS ET INTERFACE =====

    def update_performance_metrics(self):
        """Mettre à jour les métriques de performance"""
        try:
            # Cache hit rate
            cache_stats = self.cache.get_stats()
            self.cache_label.config(text=f"Cache: {cache_stats['hit_rate']:.0f}%")

            # Utilisation mémoire
            if PSUTIL_AVAILABLE:
                memory_mb = psutil.Process().memory_info().rss / 1024**2
                self.memory_label.config(text=f"RAM: {memory_mb:.0f}MB")

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
        """Scanner le dossier pour les fichiers JSON"""
        folder = self.folder_var.get()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Erreur", "Veuillez sélectionner un dossier valide")
            return

        json_files = glob.glob(os.path.join(folder, "*.json"))

        if not json_files:
            messagebox.showwarning("Attention", "Aucun fichier JSON trouvé dans ce dossier")
            return

        self.json_files = json_files

        # Analyse rapide des fichiers
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
        """Traiter les fichiers JSON avec IA avancée"""
        self.process_btn.config(state='disabled')
        self.progress_var.set(0)

        # Lancer le traitement dans un thread séparé
        thread = threading.Thread(target=self._process_files_thread)
        thread.daemon = True
        thread.start()

    def _process_files_thread(self):
        """Thread de traitement des fichiers"""
        try:
            def _process_internal():
                all_races = []
                total_files = len(self.json_files)

                self.queue.put(('progress', 5, f"Initialisation du traitement IA..."))

                for i, file_path in enumerate(self.json_files):
                    progress = (i + 1) / total_files * 70
                    filename = os.path.basename(file_path)

                    self.queue.put(('progress', progress, f"Traitement: {filename} ({i+1}/{total_files})"))

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            race_data = json.load(f)

                        # Extraction de la date
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

                # Extraction des features
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

            # Exécution avec gestion d'erreurs
            self.error_handler.robust_execute(_process_internal, 'data_loading')

        except Exception as e:
            self.queue.put(('error', f"Erreur lors du traitement IA: {str(e)}"))

    def optimize_dataframe_memory(self, df):
        """Optimiser l'usage mémoire du DataFrame"""
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

        self.error_handler.log_info(f"Optimisation mémoire: {reduction:.1f}% de réduction", "memory_optimization")

        return df

    # ===== MÉTHODES D'ENTRAÎNEMENT =====

    def train_ensemble_models(self):
        """Entraîner l'ensemble de modèles IA"""
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
        """Thread d'entraînement de l'ensemble"""
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
                    f"Validation: {cv_folds} folds temporels\n\n"
                ))

                # Sélection des features
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

                    # Entraîner l'ensemble
                    cv_scores = self.ensemble.train_ensemble(X, y, target_name, cv_folds)

                    # Afficher les résultats
                    self.queue.put(('training_info', f"📊 Résultats validation croisée:\n"))
                    for model_name, score in cv_scores.items():
                        self.queue.put(('training_info', f"  • {model_name}: {score:.4f}\n"))

                    all_results[target_name] = {
                        'cv_scores': cv_scores,
                        'weights': self.ensemble.weights.get(target_name, {}),
                        'best_score': max(cv_scores.values()) if cv_scores else 0
                    }

                # Sauvegarde des noms de features
                self.ensemble.feature_names = feature_cols
                self.ensemble.is_trained = True
                self.training_results = all_results

                self.queue.put(('training_complete',
                    "🎉 Entraînement IA Ensemble terminé avec succès !\n\n"
                    "L'intelligence artificielle est prête à prédire les courses."
                ))

            # Exécution avec gestion d'erreurs
            self.error_handler.robust_execute(_train_internal, 'model_training')

        except Exception as e:
            self.queue.put(('training_error', f"Erreur d'entraînement IA: {str(e)}"))

    def select_training_features(self, df):
        """Sélectionner les features pour l'entraînement"""
        # Features numériques de base
        base_features = [
            'age', 'nombreCourses', 'win_rate', 'place_rate', 'place2_rate', 'place3_rate',
            'top3_rate', 'recent_form_score', 'consistency_score', 'trend_score',
            'best_recent_position'
        ]

        # Features financières
        financial_features = [
            'gains_carriere', 'gains_victoires', 'gains_place', 'gains_annee_courante',
            'avg_gain_per_race', 'win_gain_ratio', 'recent_earning_trend'
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
            base_features, financial_features, market_features,
            competitive_features, interaction_features, categorical_features, temporal_features
        ]

        selected_features = []
        for group in all_feature_groups:
            for feature in group:
                if feature in df.columns:
                    selected_features.append(feature)

        return selected_features

    def check_queue(self):
        """Vérifier la queue pour les mises à jour des threads"""
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

        except queue.Empty:
            pass

        # Programmer la prochaine vérification
        self.root.after(100, self.check_queue)

    # ===== MÉTHODES PLACEHOLDER (À COMPLÉTER) =====

    def on_prediction_mode_change(self):
        """Gérer le changement de mode de prédiction"""
        mode = self.prediction_mode.get()

        if mode == "existing":
            self.existing_frame.pack(fill='x', pady=10)
            self.new_file_frame.pack_forget()
        else:
            self.existing_frame.pack_forget()
            self.new_file_frame.pack(fill='x', pady=10)

    def browse_new_race_file(self):
        """Parcourir pour nouveau fichier"""
        pass

    def load_new_race_file(self):
        """Charger nouveau fichier"""
        pass

    def predict_race(self):
        """Prédire une course"""
        pass

    def sort_treeview_column(self, col):
        """Trier une colonne du tableau"""
        pass

    def show_model_performance(self):
        """Afficher performance des modèles"""
        pass

    def show_feature_importance(self):
        """Afficher importance des features"""
        pass

    def show_temporal_analysis(self):
        """Afficher analyse temporelle"""
        pass

    def run_backtest(self):
        """Lancer le backtesting"""
        pass

    def save_models(self):
        """Sauvegarder les modèles"""
        pass

    def load_models(self):
        """Charger les modèles"""
        pass
