"""
Module d'ensemble de modèles IA avec spécialisation Galop/Trot
Combine LightGBM, XGBoost, CatBoost et Random Forest pour chaque type de course
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from error_handler import RobustErrorHandler

# Gestion de CatBoost (optionnel)
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost non disponible - utilisation de LightGBM et XGBoost uniquement")


class SpecializedHorseRacingEnsemble:
    """Ensemble de modèles IA spécialisés pour prédictions hippiques Galop/Trot"""
    
    def __init__(self):
        # Modèles organisés par [type_course][target][model_name]
        self.models = {'GALOP': {}, 'TROT': {}, 'MIXED': {}}
        self.weights = {'GALOP': {}, 'TROT': {}, 'MIXED': {}}
        self.feature_names = {'GALOP': [], 'TROT': [], 'MIXED': []}
        self.is_trained = {'GALOP': False, 'TROT': False, 'MIXED': False}
        self.error_handler = RobustErrorHandler()
        
        # Performances par type de course
        self.training_history = {
            'GALOP': {},
            'TROT': {},
            'MIXED': {}
        }

    def detect_race_types(self, data):
        """Détecter les types de courses disponibles dans les données"""
        if 'allure' not in data.columns:
            return ['MIXED']
        
        race_types = []
        for allure in ['GALOP', 'TROT']:
            if allure in data['allure'].values:
                count = len(data[data['allure'] == allure])
                if count > 0:
                    race_types.append(allure)
                    print(f"📊 {allure}: {count} chevaux disponibles pour l'entraînement")
        
        if not race_types:
            race_types = ['MIXED']
            print("⚠️ Types de course non détectés - utilisation du mode mixte")
            
        return race_types

    def create_specialized_model_configs(self, race_type, target_type='classification'):
        """Créer des configurations spécialisées par type de course"""
        base_configs = self.create_model_configs(target_type)
        
        # Ajustements spécifiques par type de course
        if race_type == 'TROT':
            # Le trot est plus prévisible - réglages plus conservateurs
            for model_name in base_configs:
                if model_name == 'lgb':
                    base_configs[model_name]['base_params']['learning_rate'] = 0.005
                    base_configs[model_name]['base_params']['num_leaves'] = 25
                elif model_name == 'xgb':
                    base_configs[model_name]['base_params']['learning_rate'] = 0.005
                    base_configs[model_name]['base_params']['max_depth'] = 4
                elif model_name == 'rf':
                    base_configs[model_name]['base_params']['n_estimators'] = 400
                    
        elif race_type == 'GALOP':
            # Le galop est plus imprévisible - réglages plus agressifs
            for model_name in base_configs:
                if model_name == 'lgb':
                    base_configs[model_name]['base_params']['learning_rate'] = 0.02
                    base_configs[model_name]['base_params']['feature_fraction'] = 0.7
                elif model_name == 'xgb':
                    base_configs[model_name]['base_params']['learning_rate'] = 0.02
                    base_configs[model_name]['base_params']['subsample'] = 0.6
        
        return base_configs

    def create_model_configs(self, target_type='classification'):
        """Créer les configurations de base des modèles"""
        configs = {}
    
        if target_type == 'classification':
            configs['lgb'] = {
                'model_class': lgb.LGBMClassifier,
                'base_params': {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 20,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
            }
    
            configs['xgb'] = {
                'model_class': xgb.XGBClassifier,
                'base_params': {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'n_estimators': 300,
                    'max_depth': 5,
                    'learning_rate': 0.01,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'random_state': 42,
                    'n_jobs': -1
                }
            }
    
            if CATBOOST_AVAILABLE:
                configs['catboost'] = {
                    'model_class': CatBoostClassifier,
                    'base_params': {
                        'iterations': 300,
                        'depth': 5,
                        'learning_rate': 0.01,
                        'random_seed': 42,
                        'verbose': False
                    }
                }
    
            configs['rf'] = {
                'model_class': RandomForestClassifier,
                'base_params': {
                    'n_estimators': 300,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            }
    
        else:  # regression
            configs['lgb'] = {
                'model_class': lgb.LGBMRegressor,
                'base_params': {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 20,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
            }
    
            configs['xgb'] = {
                'model_class': xgb.XGBRegressor,
                'base_params': {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'n_estimators': 300,
                    'max_depth': 5,
                    'learning_rate': 0.01,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'random_state': 42,
                    'n_jobs': -1
                }
            }
    
            if CATBOOST_AVAILABLE:
                configs['catboost'] = {
                    'model_class': CatBoostRegressor,
                    'base_params': {
                        'iterations': 300,
                        'depth': 5,
                        'learning_rate': 0.01,
                        'random_seed': 42,
                        'verbose': False
                    }
                }
    
            configs['rf'] = {
                'model_class': RandomForestRegressor,
                'base_params': {
                    'n_estimators': 300,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            }
    
        return configs

    def train_specialized_ensemble(self, X, y, target_name, race_type, cv_folds=5):
        """Entraîner l'ensemble spécialisé pour un type de course"""
        self.error_handler.log_info(f"Début entraînement ensemble {race_type} pour {target_name}", "ensemble_training")

        target_type = 'classification' if target_name in ['win', 'place'] else 'regression'
        model_configs = self.create_specialized_model_configs(race_type, target_type)

        if race_type not in self.models:
            self.models[race_type] = {}
        if target_name not in self.models[race_type]:
            self.models[race_type][target_name] = {}
        
        cv_scores = {}

        # Validation croisée temporelle
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        for model_name, config in model_configs.items():
            try:
                self.error_handler.log_info(f"Entraînement {model_name} pour {race_type}/{target_name}", "model_training")

                # Cross-validation
                model = config['model_class'](**config['base_params'])

                if target_type == 'classification':
                    cv_score = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=1)
                else:
                    cv_score = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=1)
                    cv_score = -cv_score

                mean_score = cv_score.mean()
                cv_scores[model_name] = mean_score

                # Entraînement final
                model.fit(X, y)
                self.models[race_type][target_name][model_name] = model

                self.error_handler.log_info(f"{model_name} {race_type} terminé: {mean_score:.4f} (±{cv_score.std():.4f})", "model_training")

            except Exception as e:
                self.error_handler.log_warning(f"Erreur {model_name} {race_type}: {str(e)}", "model_training")
                continue

        # Calcul des poids spécialisés
        if cv_scores:
            if race_type not in self.weights:
                self.weights[race_type] = {}
                
            if target_type == 'classification':
                total_score = sum(cv_scores.values())
                self.weights[race_type][target_name] = {name: score/total_score for name, score in cv_scores.items()}
            else:
                inv_scores = {name: 1/max(score, 0.001) for name, score in cv_scores.items()}
                total_inv_score = sum(inv_scores.values())
                self.weights[race_type][target_name] = {name: score/total_inv_score for name, score in inv_scores.items()}

        # Sauvegarder l'historique
        if race_type not in self.training_history:
            self.training_history[race_type] = {}
        self.training_history[race_type][target_name] = {
            'cv_scores': cv_scores,
            'best_score': max(cv_scores.values()) if cv_scores else 0,
            'model_count': len(cv_scores)
        }

        self.error_handler.log_info(f"Entraînement ensemble {race_type} terminé pour {target_name}", "ensemble_training")
        return cv_scores

    def train_all_race_types(self, data, feature_engineer, cv_folds=5):
        """Entraîner tous les modèles pour tous les types de course disponibles"""
        race_types = self.detect_race_types(data)
        all_results = {}
        
        for race_type in race_types:
            print(f"\n🎯 ENTRAÎNEMENT MODÈLES {race_type}")
            print("=" * 50)
            
            # Filtrer les données par type de course
            if race_type == 'MIXED':
                race_data = data.copy()
            else:
                race_data = data[data['allure'] == race_type].copy()
            
            if len(race_data) < 50:
                print(f"⚠️ Pas assez de données pour {race_type} ({len(race_data)} chevaux) - ignoré")
                continue
            
            # Obtenir les features spécialisées
            feature_list = feature_engineer.get_feature_list_by_type(race_type)
            available_features = [f for f in feature_list if f in race_data.columns]
            
            if len(available_features) < 10:
                print(f"⚠️ Pas assez de features pour {race_type} ({len(available_features)}) - ignoré")
                continue
            
            X = race_data[available_features].fillna(0)
            self.feature_names[race_type] = available_features
            
            # Variables cibles
            targets = {
                'win': race_data['won_race'],
                'place': race_data['top3_finish'],
                'position': race_data['final_position']
            }
            
            race_results = {}
            
            for target_name, y in targets.items():
                if y.sum() == 0:  # Pas de données positives
                    print(f"⚠️ Pas de données positives pour {race_type}/{target_name} - ignoré")
                    continue
                    
                cv_scores = self.train_specialized_ensemble(X, y, target_name, race_type, cv_folds)
                race_results[target_name] = cv_scores
            
            if race_results:
                all_results[race_type] = race_results
                self.is_trained[race_type] = True
                print(f"✅ {race_type} entraîné avec succès")
            else:
                print(f"❌ Échec entraînement {race_type}")
        
        return all_results

    def predict_specialized_ensemble(self, X, target_name, race_type):
        """Prédiction spécialisée par type de course"""
        if race_type not in self.models or target_name not in self.models[race_type]:
            raise ValueError(f"Aucun modèle entraîné pour {race_type}/{target_name}")
        
        if not self.models[race_type][target_name]:
            raise ValueError(f"Aucun modèle disponible pour {race_type}/{target_name}")

        predictions = []
        weights = []

        for model_name, model in self.models[race_type][target_name].items():
            try:
                if hasattr(model, 'predict_proba') and target_name in ['win', 'place']:
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)

                predictions.append(pred)
                weight = self.weights[race_type][target_name].get(model_name, 1.0)
                weights.append(weight)

            except Exception as e:
                self.error_handler.log_warning(f"Erreur prédiction {model_name} {race_type}: {str(e)}", "prediction")
                continue

        if not predictions:
            raise ValueError(f"Aucune prédiction disponible pour {race_type}/{target_name}")

        # Moyenne pondérée
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()

        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred

    def predict_with_race_type_detection(self, X, target_name, data=None):
        """Prédiction avec détection automatique du type de course"""
        if data is not None and 'allure' in data.columns:
            # Détecter le type de course majoritaire
            race_types = data['allure'].value_counts()
            if len(race_types) > 0:
                main_race_type = race_types.index[0]
                
                if main_race_type in self.models and self.is_trained[main_race_type]:
                    return self.predict_specialized_ensemble(X, target_name, main_race_type)
        
        # Fallback: essayer les modèles disponibles
        for race_type in ['GALOP', 'TROT', 'MIXED']:
            if self.is_trained[race_type] and target_name in self.models.get(race_type, {}):
                try:
                    return self.predict_specialized_ensemble(X, target_name, race_type)
                except:
                    continue
        
        raise ValueError(f"Aucun modèle compatible trouvé pour {target_name}")

    def get_specialized_feature_importance(self, target_name, race_type):
        """Obtenir l'importance des features pour un type de course spécifique"""
        if race_type not in self.models or target_name not in self.models[race_type]:
            return None

        importances = {}
        
        for model_name, model in self.models[race_type][target_name].items():
            if hasattr(model, 'feature_importances_'):
                importances[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances[model_name] = np.abs(model.coef_)

        return importances

    def get_training_summary(self):
        """Obtenir un résumé complet de l'entraînement"""
        summary = {
            'race_types_trained': [rt for rt, trained in self.is_trained.items() if trained],
            'total_models': 0,
            'performance_by_type': {}
        }
        
        for race_type, trained in self.is_trained.items():
            if trained:
                type_models = 0
                type_performance = {}
                
                for target_name, target_models in self.models[race_type].items():
                    type_models += len(target_models)
                    
                    if race_type in self.training_history and target_name in self.training_history[race_type]:
                        type_performance[target_name] = self.training_history[race_type][target_name]['best_score']
                
                summary['total_models'] += type_models
                summary['performance_by_type'][race_type] = {
                    'model_count': type_models,
                    'performance': type_performance
                }
        
        return summary

    def get_best_race_type_for_prediction(self, data):
        """Déterminer le meilleur type de modèle pour une prédiction"""
        if 'allure' not in data.columns:
            # Pas d'info sur l'allure, utiliser le modèle le plus performant
            best_type = None
            best_score = 0
            
            for race_type in ['GALOP', 'TROT', 'MIXED']:
                if self.is_trained[race_type]:
                    if race_type in self.training_history and 'win' in self.training_history[race_type]:
                        score = self.training_history[race_type]['win']['best_score']
                        if score > best_score:
                            best_score = score
                            best_type = race_type
            
            return best_type or 'MIXED'
        
        # Utiliser l'allure détectée
        race_type = data['allure'].iloc[0] if len(data) > 0 else 'MIXED'
        
        if race_type in self.is_trained and self.is_trained[race_type]:
            return race_type
        
        return 'MIXED'

    def calculate_specialized_confidence(self, X, target_name, race_type):
        """Calculer la confiance spécialisée par type de course"""
        if race_type not in self.models or target_name not in self.models[race_type]:
            return np.ones(len(X)) * 0.5

        individual_preds = {}
        
        for model_name, model in self.models[race_type][target_name].items():
            try:
                if hasattr(model, 'predict_proba') and target_name in ['win', 'place']:
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                individual_preds[model_name] = pred
            except Exception as e:
                self.error_handler.log_warning(f"Erreur calcul confiance {model_name} {race_type}: {str(e)}", "confidence")
                continue
        
        if len(individual_preds) < 2:
            return np.ones(len(X)) * 0.5

        predictions_array = np.array(list(individual_preds.values()))
        
        # Confiance = 1 - variance normalisée
        variance = np.var(predictions_array, axis=0)
        max_variance = np.max(variance) if np.max(variance) > 0 else 1
        confidence = 1 - (variance / max_variance)
        
        return confidence

    def reset_race_type(self, race_type):
        """Réinitialiser un type de course spécifique"""
        if race_type in self.models:
            self.models[race_type] = {}
        if race_type in self.weights:
            self.weights[race_type] = {}
        if race_type in self.feature_names:
            self.feature_names[race_type] = []
        if race_type in self.is_trained:
            self.is_trained[race_type] = False
        if race_type in self.training_history:
            self.training_history[race_type] = {}
        
        self.error_handler.log_info(f"Type de course {race_type} réinitialisé", "ensemble_reset")

    def reset_all(self):
        """Réinitialiser complètement l'ensemble"""
        for race_type in ['GALOP', 'TROT', 'MIXED']:
            self.reset_race_type(race_type)
        
        self.error_handler.log_info("Ensemble complet réinitialisé", "ensemble_reset")