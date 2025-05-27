"""
Méthodes complémentaires pour l'interface graphique
Complète les fonctionnalités de prédiction, analytics et backtesting
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import json
import os
import re
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns


class GUIMethods:
    """Méthodes complémentaires pour l'interface graphique"""
    
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
        """Charger un nouveau fichier de course"""
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

            # Extraction des features
            print(f"🔄 Traitement du fichier: {basename}")
            print(f"📊 Participants trouvés: {len(participants)}")

            self.new_race_data = self.feature_engineer.extract_comprehensive_features(temp_df)

            num_horses = len(self.new_race_data)
            num_features = self.new_race_data.shape[1]

            # Vérifier s'il y a de nouvelles valeurs
            unknown_report = self.feature_engineer.get_unknown_values_report()

            if unknown_report:
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
                messagebox.showinfo("Succès",
                    f"✅ Fichier chargé parfaitement !\n\n"
                    f"• {num_horses} chevaux traités par l'IA\n"
                    f"• {num_features} caractéristiques extraites\n"
                    f"• Tous les participants sont connus de l'IA\n"
                    f"• Prêt pour prédiction optimale"
                )
                self.new_file_info.config(text=f"✅ Fichier chargé: {num_horses} chevaux - {basename}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement: {str(e)}")
            self.new_file_info.config(text="❌ Erreur de chargement")

    def predict_race(self):
        """Prédire une course avec l'ensemble IA"""
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

                    # Calcul de confiance
                    confidence = self.ensemble.calculate_prediction_confidence(X_pred, target_name)
                    confidence_scores[target_name] = confidence

                except Exception as e:
                    self.error_handler.log_warning(f"Erreur prédiction {target_name}: {str(e)}", "prediction")
                    continue

            if not predictions:
                messagebox.showerror("Erreur", "Aucune prédiction n'a pu être générée")
                return

            # Préparation des résultats
            results_df = race_data[['numPmu', 'nom', 'driver', 'direct_odds']].copy()

            if 'position' in predictions:
                results_df['pred_position'] = predictions['position']
                results_df['predicted_rank'] = results_df['pred_position'].rank()
            else:
                results_df['predicted_rank'] = range(1, len(results_df) + 1)

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

            # Tri par rang prédit
            results_df = results_df.sort_values('predicted_rank')

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
                    i + 1,
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

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la prédiction: {str(e)}")

    def generate_ai_recommendations(self, results_df):
        """Générer les recommandations IA avancées"""
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

        outsiders = results_df[(results_df['direct_odds'] > 10) &
                              (results_df['prob_win'] > 0.05)]

        if not outsiders.empty:
            best_outsider = outsiders.loc[outsiders['prob_win'].idxmax()]
            expected_value = best_outsider['prob_win'] * best_outsider['direct_odds']

            if expected_value > 1.2:
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
        """Trier le tableau par colonne"""
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

    def show_model_performance(self):
        """Afficher les performances des modèles"""
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

        # Graphique 4: Importance des features (si disponible)
        if hasattr(self.ensemble, 'feature_names') and self.ensemble.feature_names:
            # Essayer d'obtenir l'importance des features
            try:
                target_name = list(self.ensemble.models.keys())[0]
                model_name = list(self.ensemble.models[target_name].keys())[0]
                model = self.ensemble.models[target_name][model_name]
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
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
        """Afficher l'importance des features"""
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
        """Afficher l'analyse temporelle"""
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

    def run_backtest(self):
        """Lancer le backtesting"""
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
        """Thread de backtesting"""
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

    def save_models(self):
        """Sauvegarder l'ensemble de modèles"""
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
                import pickle
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
        """Charger l'ensemble de modèles"""
        filename = filedialog.askopenfilename(
            title="Charger l'IA Ensemble",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if filename:
            try:
                import pickle
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

    def enhanced_check_queue(self):
        """Version améliorée de check_queue avec gestion des backtests"""
        try:
            while True:
                msg_type, *args = self.queue.get_nowait()

                # Gestion des nouveaux messages de backtest
                if msg_type == 'backtest_results':
                    report = args[0]
                    self.backtest_text.delete(1.0, tk.END)
                    self.backtest_text.insert(1.0, report)

                elif msg_type == 'backtest_error':
                    error_msg = args[0]
                    messagebox.showerror("Erreur Backtest", error_msg)
                    self.backtest_text.insert(tk.END, f"\n❌ ERREUR: {error_msg}\n")

                # Appeler la méthode check_queue originale pour les autres messages
                else:
                    # Remettre le message dans la queue pour traitement par check_queue
                    self.queue.put((msg_type, *args))
                    break

        except:
            pass
