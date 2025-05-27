"""
Module de backtesting et validation des performances
Teste les stratégies de pari et valide les performances des modèles IA
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime
from error_handler import RobustErrorHandler


class BacktestingEngine:
    """Moteur de backtesting pour valider les performances des modèles"""
    
    def __init__(self, ensemble_model=None):
        self.ensemble_model = ensemble_model
        self.error_handler = RobustErrorHandler()
        self.results_history = []
        
    def split_data_temporal(self, data: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Diviser les données de façon temporelle"""
        if 'race_date' not in data.columns:
            raise ValueError("Colonne 'race_date' requise pour le split temporel")
            
        # Trier par date
        data_sorted = data.sort_values('race_date')
        
        # Point de division
        split_idx = int(len(data_sorted) * (1 - test_ratio))
        
        train_data = data_sorted.iloc[:split_idx].copy()
        test_data = data_sorted.iloc[split_idx:].copy()
        
        self.error_handler.log_info(f"Split temporel: {len(train_data)} train, {len(test_data)} test", "backtesting")
        
        return train_data, test_data
    
    def run_backtest(self, data: pd.DataFrame, strategy: str, test_period: float = 0.2, 
                     feature_names: List[str] = None) -> Dict[str, Any]:
        """Lancer un backtest complet"""
        
        if self.ensemble_model is None or not self.ensemble_model.is_trained:
            raise ValueError("Modèle ensemble requis et entraîné")
            
        self.error_handler.log_info(f"Début backtest - Stratégie: {strategy}", "backtesting")
        
        # Diviser les données
        train_data, test_data = self.split_data_temporal(data, test_period)
        
        # Obtenir les courses uniques pour le test
        unique_races = test_data['race_file'].unique()
        total_races = len(unique_races)
        
        if total_races == 0:
            raise ValueError("Aucune course de test disponible")
        
        # Initialiser les résultats
        results = {
            'strategy': strategy,
            'total_races': total_races,
            'test_period': test_period,
            'correct_predictions': 0,
            'total_bets': 0,
            'winning_bets': 0,
            'total_roi': 0,
            'race_results': [],
            'place_stats': {
                'first_choice': {'wins': 0, 'top3': 0, 'total': 0},
                'second_choice': {'wins': 0, 'top3': 0, 'total': 0}
            },
            'performance_metrics': {}
        }
        
        # Traiter chaque course
        for i, race_file in enumerate(unique_races):
            try:
                race_result = self._process_single_race(test_data, race_file, strategy, feature_names)
                
                if race_result:
                    results['race_results'].append(race_result)
                    
                    # Mise à jour des totaux
                    results['total_bets'] += race_result.get('num_bets', 0)
                    results['winning_bets'] += race_result.get('winning_bets', 0)
                    results['total_roi'] += race_result.get('roi', 0)
                    
                    if race_result.get('predicted_winner_correct', False):
                        results['correct_predictions'] += 1
                    
                    # Mise à jour des stats place
                    if 'place_stats' in race_result:
                        for choice in ['first_choice', 'second_choice']:
                            if choice in race_result['place_stats']:
                                for metric in ['wins', 'top3', 'total']:
                                    results['place_stats'][choice][metric] += race_result['place_stats'][choice].get(metric, 0)
                
            except Exception as e:
                self.error_handler.log_warning(f"Erreur course {race_file}: {str(e)}", "backtesting")
                continue
            
            # Log de progression
            if (i + 1) % 50 == 0:  # Tous les 50 courses
                progress = (i + 1) / total_races * 100
                self.error_handler.log_info(f"Progression: {progress:.1f}% ({i+1}/{total_races})", "backtesting")
        
        # Calculer les métriques finales
        results['performance_metrics'] = self._calculate_final_metrics(results)
        
        # Sauvegarder dans l'historique
        results['timestamp'] = datetime.now().isoformat()
        self.results_history.append(results)
        
        self.error_handler.log_info("Backtest terminé", "backtesting")
        
        return results
    
    def _process_single_race(self, test_data, race_file, strategy, feature_names):
        """Version avec debug pour identifier le problème"""
        
        race_data = test_data[test_data['race_file'] == race_file].copy()
        
        if len(race_data) < 3:
            return None
        
        print(f"\n🔍 DEBUG - Course: {race_file}")
        print(f"   Chevaux: {len(race_data)}")
        
        # DIAGNOSTIC 1: Vérifier les variables cibles
        if 'final_position' in race_data.columns:
            positions = race_data['final_position'].value_counts().sort_index()
            print(f"   Positions: {dict(positions)}")
            
            winners = race_data[race_data['final_position'] == 1]
            print(f"   Gagnants trouvés: {len(winners)}")
            
            if len(winners) == 0:
                print("   ❌ PROBLÈME: Aucun gagnant trouvé!")
                return None
        else:
            print("   ❌ PROBLÈME: Colonne 'final_position' manquante!")
            return None
        
        # DIAGNOSTIC 2: Vérifier les features
        if feature_names:
            missing_features = [f for f in feature_names if f not in race_data.columns]
            if missing_features:
                print(f"   ❌ Features manquantes: {len(missing_features)}/{len(feature_names)}")
                return None
            else:
                print(f"   ✅ Features OK: {len(feature_names)}")
        
        # DIAGNOSTIC 3: Tester les prédictions
        try:
            X_race = race_data[feature_names].fillna(0) if feature_names else race_data.fillna(0)
            
            # Détecter le type de course
            race_type = self.ensemble_model.get_best_race_type_for_prediction(race_data)
            print(f"   Type détecté: {race_type}")
            
            # Test prédiction
            win_probs = self.ensemble_model.predict_specialized_ensemble(X_race, 'win', race_type)
            place_probs = self.ensemble_model.predict_specialized_ensemble(X_race, 'place', race_type)
            
            print(f"   Prédictions win: min={win_probs.min():.3f}, max={win_probs.max():.3f}")
            print(f"   Prédictions place: min={place_probs.min():.3f}, max={place_probs.max():.3f}")
            
            race_data['pred_win_prob'] = win_probs
            race_data['pred_place_prob'] = place_probs
            
            # Test de la stratégie
            if strategy == 'place_strategy':
                race_data_sorted = race_data.sort_values('pred_place_prob', ascending=False)
                result = self._apply_place_betting_strategy(race_data_sorted)
            else:
                race_data_sorted = race_data.sort_values('pred_win_prob', ascending=False)
                result = self._apply_betting_strategy(race_data_sorted, strategy)
            
            if result:
                print(f"   Résultat: {result}")
            else:
                print("   ❌ Aucun résultat de stratégie")
                
            return result
                
        except Exception as e:
            print(f"   ❌ Erreur prédiction: {str(e)}")
            return None
        
    def _apply_place_betting_strategy(self, race_data: pd.DataFrame) -> Dict[str, Any]:
        """Stratégie basée sur les probabilités de place"""
        results = {
            'num_bets': 0,
            'winning_bets': 0,
            'roi': 0,
            'predicted_winner_correct': False,
            'place_stats': {
                'first_choice': {'wins': 0, 'top3': 0, 'total': 0},
                'second_choice': {'wins': 0, 'top3': 0, 'total': 0}
            }
        }
        
        # Vérifier les résultats réels
        if 'final_position' not in race_data.columns:
            return results
        
        race_data_with_positions = race_data.dropna(subset=['final_position'])
        if len(race_data_with_positions) == 0:
            return results
        
        # Premier et deuxième choix
        if len(race_data) >= 2:
            first_choice = race_data.iloc[0]
            second_choice = race_data.iloc[1]
            
            # Stats premier choix
            first_position = first_choice['final_position']
            results['place_stats']['first_choice']['total'] = 1
            
            if first_position == 1:
                results['place_stats']['first_choice']['wins'] = 1
                results['place_stats']['first_choice']['top3'] = 1
                results['predicted_winner_correct'] = True
            elif first_position <= 3:
                results['place_stats']['first_choice']['top3'] = 1
            
            # Stats deuxième choix
            second_position = second_choice['final_position']
            results['place_stats']['second_choice']['total'] = 1
            
            if second_position == 1:
                results['place_stats']['second_choice']['wins'] = 1
                results['place_stats']['second_choice']['top3'] = 1
            elif second_position <= 3:
                results['place_stats']['second_choice']['top3'] = 1
            
            # Pari simple sur le premier choix
            results['num_bets'] = 1
            if first_position == 1:
                results['winning_bets'] = 1
                # ROI basique sans calcul de cotes complexe
                results['roi'] = 0
        
        return results
    
    def _apply_betting_strategy(self, race_data: pd.DataFrame, strategy: str) -> Dict[str, Any]:
        """Appliquer une stratégie de pari classique"""
        results = {
            'num_bets': 0,
            'winning_bets': 0,
            'roi': 0,
            'predicted_winner_correct': False
        }
        
        # Vérifier les résultats réels
        if 'final_position' not in race_data.columns:
            return results
        
        actual_winner = race_data[race_data['final_position'] == 1]
        if len(actual_winner) == 0:
            return results
        
        predicted_winner = race_data.iloc[0]  # Premier dans le classement prédit
        
        # Vérifier si la prédiction est correcte
        results['predicted_winner_correct'] = (
            predicted_winner['numPmu'] == actual_winner.iloc[0]['numPmu']
        )
        
        # Appliquer la stratégie spécifique
        if strategy == 'confidence':
            # Parier seulement si confiance > 60%
            if predicted_winner['pred_win_prob'] > 0.6:
                results['num_bets'] = 1
                if results['predicted_winner_correct']:
                    results['winning_bets'] = 1
                    # ROI basé sur les cotes
                    odds = predicted_winner.get('direct_odds', 2.0)
                    results['roi'] = (odds - 1) * 100
                else:
                    results['roi'] = -100
        
        elif strategy == 'top_pick':
            # Toujours parier sur le favori IA
            results['num_bets'] = 1
            if results['predicted_winner_correct']:
                results['winning_bets'] = 1
                odds = predicted_winner.get('direct_odds', 2.0)
                results['roi'] = (odds - 1) * 100
            else:
                results['roi'] = -100
        
        elif strategy == 'value_betting':
            # Parier si valeur attendue > 1.2
            odds = predicted_winner.get('direct_odds', 2.0)
            expected_value = predicted_winner['pred_win_prob'] * odds
            
            if expected_value > 1.2:
                results['num_bets'] = 1
                if results['predicted_winner_correct']:
                    results['winning_bets'] = 1
                    results['roi'] = (odds - 1) * 100
                else:
                    results['roi'] = -100
        
        return results
    
    def _calculate_final_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculer les métriques finales de performance"""
        metrics = {}
        
        # Précision des prédictions
        metrics['accuracy'] = (results['correct_predictions'] / max(results['total_races'], 1)) * 100
        
        # Taux de réussite des paris
        metrics['hit_rate'] = (results['winning_bets'] / max(results['total_bets'], 1)) * 100
        
        # ROI moyen par course
        metrics['avg_roi_per_race'] = results['total_roi'] / max(results['total_races'], 1)
        
        # ROI total
        metrics['total_roi'] = results['total_roi']
        
        # Statistiques place (si disponibles)
        if results['place_stats']['first_choice']['total'] > 0:
            first_stats = results['place_stats']['first_choice']
            metrics['first_choice_win_rate'] = (first_stats['wins'] / first_stats['total']) * 100
            metrics['first_choice_place_rate'] = (first_stats['top3'] / first_stats['total']) * 100
        
        if results['place_stats']['second_choice']['total'] > 0:
            second_stats = results['place_stats']['second_choice']
            metrics['second_choice_win_rate'] = (second_stats['wins'] / second_stats['total']) * 100
            metrics['second_choice_place_rate'] = (second_stats['top3'] / second_stats['total']) * 100
        
        # Évaluation qualitative
        accuracy = metrics['accuracy']
        if accuracy > 25:
            metrics['performance_rating'] = "EXCELLENTE"
        elif accuracy > 20:
            metrics['performance_rating'] = "BONNE"
        elif accuracy > 15:
            metrics['performance_rating'] = "CORRECTE"
        else:
            metrics['performance_rating'] = "À AMÉLIORER"
        
        # Rentabilité
        avg_roi = metrics['avg_roi_per_race']
        if avg_roi > 5:
            metrics['profitability_rating'] = "RENTABLE"
        elif avg_roi > 0:
            metrics['profitability_rating'] = "ÉQUILIBRÉE"
        else:
            metrics['profitability_rating'] = "À OPTIMISER"
        
        return metrics
    
    def compare_strategies(self, data: pd.DataFrame, strategies: List[str], 
                          test_period: float = 0.2, feature_names: List[str] = None) -> Dict[str, Any]:
        """Comparer plusieurs stratégies"""
        
        self.error_handler.log_info(f"Comparaison de {len(strategies)} stratégies", "strategy_comparison")
        
        comparison_results = {
            'strategies': {},
            'summary': {},
            'best_strategy': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Tester chaque stratégie
        for strategy in strategies:
            try:
                results = self.run_backtest(data, strategy, test_period, feature_names)
                comparison_results['strategies'][strategy] = results
                
            except Exception as e:
                self.error_handler.log_warning(f"Erreur stratégie {strategy}: {str(e)}", "strategy_comparison")
                continue
        
        # Analyser les résultats
        if comparison_results['strategies']:
            comparison_results['summary'] = self._analyze_strategy_comparison(comparison_results['strategies'])
        
        return comparison_results
    
    def _analyze_strategy_comparison(self, strategies_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyser la comparaison des stratégies"""
        
        summary = {
            'best_accuracy': {'strategy': None, 'value': 0},
            'best_roi': {'strategy': None, 'value': float('-inf')},
            'best_hit_rate': {'strategy': None, 'value': 0},
            'strategies_ranking': []
        }
        
        strategy_scores = {}
        
        for strategy_name, results in strategies_results.items():
            metrics = results.get('performance_metrics', {})
            
            accuracy = metrics.get('accuracy', 0)
            roi = metrics.get('avg_roi_per_race', 0)
            hit_rate = metrics.get('hit_rate', 0)
            
            # Trouver les meilleurs
            if accuracy > summary['best_accuracy']['value']:
                summary['best_accuracy'] = {'strategy': strategy_name, 'value': accuracy}
            
            if roi > summary['best_roi']['value']:
                summary['best_roi'] = {'strategy': strategy_name, 'value': roi}
            
            if hit_rate > summary['best_hit_rate']['value']:
                summary['best_hit_rate'] = {'strategy': strategy_name, 'value': hit_rate}
            
            # Score combiné (pondéré)
            combined_score = (accuracy * 0.4) + (max(roi, 0) * 0.4) + (hit_rate * 0.2)
            strategy_scores[strategy_name] = combined_score
        
        # Classement
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        summary['strategies_ranking'] = [
            {'strategy': name, 'score': score} for name, score in sorted_strategies
        ]
        
        return summary
    
    def get_backtest_report(self, results: Dict[str, Any]) -> str:
        """Générer un rapport de backtest formaté"""
        
        report = f"📈 RAPPORT DE BACKTESTING\n"
        report += "=" * 60 + "\n\n"
        
        # Informations générales
        report += f"🎯 Stratégie: {results.get('strategy', 'N/A')}\n"
        report += f"📊 Période de test: {results.get('test_period', 0)*100:.0f}%\n"
        report += f"🏁 Courses testées: {results.get('total_races', 0)}\n"
        report += f"📅 Date: {results.get('timestamp', 'N/A')[:10]}\n\n"
        
        # Métriques de performance
        metrics = results.get('performance_metrics', {})
        
        report += f"📊 MÉTRIQUES DE PERFORMANCE\n"
        report += "-" * 40 + "\n"
        report += f"🎯 Précision prédictions: {metrics.get('accuracy', 0):.1f}%\n"
        report += f"✅ Taux réussite paris: {metrics.get('hit_rate', 0):.1f}%\n"
        report += f"💰 ROI moyen/course: {metrics.get('avg_roi_per_race', 0):.2f}%\n"
        report += f"💎 ROI total: {metrics.get('total_roi', 0):.2f}%\n"
        report += f"🎲 Total paris: {results.get('total_bets', 0)}\n"
        report += f"🏆 Paris gagnants: {results.get('winning_bets', 0)}\n\n"
        
        # Évaluation
        report += f"⭐ Performance: {metrics.get('performance_rating', 'N/A')}\n"
        report += f"💹 Rentabilité: {metrics.get('profitability_rating', 'N/A')}\n\n"
        
        # Statistiques place (si disponibles)
        place_stats = results.get('place_stats', {})
        if place_stats.get('first_choice', {}).get('total', 0) > 0:
            report += f"🥇 STATISTIQUES PREMIER CHOIX\n"
            report += "-" * 30 + "\n"
            first = place_stats['first_choice']
            report += f"• Victoires: {first['wins']}/{first['total']} ({metrics.get('first_choice_win_rate', 0):.1f}%)\n"
            report += f"• Top 3: {first['top3']}/{first['total']} ({metrics.get('first_choice_place_rate', 0):.1f}%)\n\n"
        
        return report
    
    def clear_history(self):
        """Vider l'historique des backtests"""
        self.results_history.clear()
        self.error_handler.log_info("Historique backtesting vidé", "backtesting")
    
    def get_history_summary(self) -> Dict[str, Any]:
        """Résumé de l'historique des backtests"""
        if not self.results_history:
            return {'total_tests': 0, 'strategies_tested': []}
        
        strategies_tested = list(set([r.get('strategy') for r in self.results_history]))
        
        return {
            'total_tests': len(self.results_history),
            'strategies_tested': strategies_tested,
            'latest_test': self.results_history[-1].get('timestamp', 'N/A')
        }
