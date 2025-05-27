"""
Module d'ensemble de modèles IA
Combine LightGBM, XGBoost, CatBoost et Random Forest pour des prédictions optimales
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from error_handler import RobustErrorHandler

# Gestion de CatBoost (optionnel)
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost non disponible - utilisation de LightGBM et XGBoost uniquement")


class HorseRacingEnsemble:
    """Ensemble de modèles IA pour prédictions hippiques"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.feature_names = []
        self.is_trained = False
        self.error_handler = RobustErrorHandler()

    def create_model_configs(self, target_type='classification'):
        """Créer les configurations des modèles"""
        configs = {}
    
        if target_type == 'classification':
            # Configuration LightGBM
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
    
            # Configuration XGBoost
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
    
            # Configuration CatBoost (si disponible)
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
    
            # Configuration Random Forest
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
            # Configuration LightGBM
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
    
            # Configuration XGBoost
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
    
            # Configuration CatBoost (si disponible)
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
    
            # Configuration Random Forest
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
    
    def optimize_model_params(self, X_train, y_train, target_type='classification'):
        """Optimiser les paramètres des modèles avec GridSearchCV"""
        configs = self.create_model_configs(target_type)
        optimized_configs = {}
    
        for model_name, config in configs.items():
            self.error_handler.log_info(f"Optimisation des paramètres pour {model_name}", "model_optimization")
    
            # Définir les paramètres à optimiser
            param_grids = {
                'lgb': {
                    'num_leaves': [31, 50, 100],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'feature_fraction': [0.7, 0.8, 0.9],
                    'bagging_fraction': [0.7, 0.8, 0.9]
                },
                'xgb': {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                },
                'catboost': {
                    'depth': [4, 5, 6],
                    'learning_rate': [0.01, 0.05, 0.1]
                },
                'rf': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }

            param_grid = param_grids.get(model_name, {})
            
            if not param_grid:
                # Pas d'optimisation pour ce modèle
                optimized_configs[model_name] = config
                continue
    
            try:
                # Créer le modèle
                model = config['model_class'](**config['base_params'])
    
                # GridSearchCV pour optimiser
                scoring = 'roc_auc' if target_type == 'classification' else 'neg_root_mean_squared_error'
                grid_search = GridSearchCV(
                    estimator=model, 
                    param_grid=param_grid, 
                    cv=3,  # Réduire pour accélérer
                    scoring=scoring, 
                    n_jobs=-1, 
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
    
                self.error_handler.log_info(f"Meilleurs paramètres {model_name}: {grid_search.best_params_}", "model_optimization")
                self.error_handler.log_info(f"Meilleur score {model_name}: {grid_search.best_score_:.4f}", "model_optimization")
    
                # Mettre à jour les configurations
                optimized_configs[model_name] = {
                    'model_class': config['model_class'],
                    'base_params': {**config['base_params'], **grid_search.best_params_}
                }
                
            except Exception as e:
                self.error_handler.log_warning(f"Échec optimisation {model_name}: {str(e)}", "model_optimization")
                # Utiliser la config de base
                optimized_configs[model_name] = config
    
        return optimized_configs

    def train_ensemble(self, X, y, target_name, cv_folds=5):
        """Entraîner l'ensemble de modèles avec validation croisée"""
        self.error_handler.log_info(f"Début entraînement ensemble pour {target_name}", "ensemble_training")

        target_type = 'classification' if target_name in ['win', 'place'] else 'regression'
        model_configs = self.create_model_configs(target_type)

        self.models[target_name] = {}
        cv_scores = {}

        # Validation croisée temporelle
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        for model_name, config in model_configs.items():
            try:
                self.error_handler.log_info(f"Entraînement {model_name} pour {target_name}", "model_training")

                # Cross-validation
                model = config['model_class'](**config['base_params'])

                if target_type == 'classification':
                    cv_score = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=1)
                else:
                    cv_score = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=1)
                    cv_score = -cv_score  # Inverser pour avoir des scores positifs

                mean_score = cv_score.mean()
                cv_scores[model_name] = mean_score

                # Entraînement final sur tout le dataset
                model.fit(X, y)
                self.models[target_name][model_name] = model

                self.error_handler.log_info(f"{model_name} terminé: {mean_score:.4f} (±{cv_score.std():.4f})", "model_training")

            except Exception as e:
                self.error_handler.log_warning(f"Erreur {model_name}: {str(e)}", "model_training")
                continue

        # Calcul des poids basé sur les performances
        if cv_scores:
            if target_type == 'classification':
                # Pour classification, plus haut = meilleur
                total_score = sum(cv_scores.values())
                self.weights[target_name] = {name: score/total_score for name, score in cv_scores.items()}
            else:
                # Pour régression, plus bas = meilleur, donc inverser
                inv_scores = {name: 1/max(score, 0.001) for name, score in cv_scores.items()}
                total_inv_score = sum(inv_scores.values())
                self.weights[target_name] = {name: score/total_inv_score for name, score in inv_scores.items()}

        self.error_handler.log_info(f"Entraînement ensemble terminé pour {target_name}", "ensemble_training")
        return cv_scores

    def predict_ensemble(self, X, target_name):
        """Prédiction ensemble pondérée"""
        if target_name not in self.models or not self.models[target_name]:
            raise ValueError(f"Aucun modèle entraîné pour {target_name}")

        predictions = []
        weights = []

        for model_name, model in self.models[target_name].items():
            try:
                if hasattr(model, 'predict_proba') and target_name in ['win', 'place']:
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)

                predictions.append(pred)
                weights.append(self.weights[target_name].get(model_name, 1.0))

            except Exception as e:
                self.error_handler.log_warning(f"Erreur prédiction {model_name}: {str(e)}", "prediction")
                continue

        if not predictions:
            raise ValueError(f"Aucune prédiction disponible pour {target_name}")

        # Moyenne pondérée
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normaliser

        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        return ensemble_pred

    def get_feature_importance(self, target_name):
        """Obtenir l'importance des features pour un target donné"""
        if target_name not in self.models or not self.models[target_name]:
            return None

        importances = {}
        
        for model_name, model in self.models[target_name].items():
            if hasattr(model, 'feature_importances_'):
                importances[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances[model_name] = np.abs(model.coef_)

        return importances

    def get_model_predictions(self, X, target_name):
        """Obtenir les prédictions individuelles de chaque modèle"""
        if target_name not in self.models or not self.models[target_name]:
            return {}

        individual_predictions = {}

        for model_name, model in self.models[target_name].items():
            try:
                if hasattr(model, 'predict_proba') and target_name in ['win', 'place']:
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                individual_predictions[model_name] = pred
            except Exception as e:
                self.error_handler.log_warning(f"Erreur prédiction individuelle {model_name}: {str(e)}", "prediction")
                continue

        return individual_predictions

    def calculate_prediction_confidence(self, X, target_name):
        """Calculer la confiance des prédictions basée sur la variance"""
        individual_preds = self.get_model_predictions(X, target_name)
        
        if len(individual_preds) < 2:
            return np.ones(len(X)) * 0.5  # Confiance moyenne si un seul modèle

        predictions_array = np.array(list(individual_preds.values()))
        
        # Confiance = 1 - variance normalisée
        variance = np.var(predictions_array, axis=0)
        max_variance = np.max(variance) if np.max(variance) > 0 else 1
        confidence = 1 - (variance / max_variance)
        
        return confidence

    def get_ensemble_summary(self):
        """Obtenir un résumé de l'ensemble"""
        summary = {
            'is_trained': self.is_trained,
            'targets': list(self.models.keys()),
            'models_per_target': {},
            'total_models': 0,
            'feature_count': len(self.feature_names)
        }
        
        for target_name, models in self.models.items():
            summary['models_per_target'][target_name] = list(models.keys())
            summary['total_models'] += len(models)
            
        return summary

    def validate_features(self, X):
        """Valider que les features correspondent à celles d'entraînement"""
        if not self.feature_names:
            raise ValueError("Modèle non entraîné - aucune feature de référence")
            
        if isinstance(X, pd.DataFrame):
            missing_features = [f for f in self.feature_names if f not in X.columns]
            extra_features = [f for f in X.columns if f not in self.feature_names]
        else:
            # Assume numpy array
            if X.shape[1] != len(self.feature_names):
                raise ValueError(f"Nombre de features incorrect: {X.shape[1]} vs {len(self.feature_names)} attendues")
            return True
            
        if missing_features:
            raise ValueError(f"Features manquantes: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            
        if extra_features:
            self.error_handler.log_warning(f"Features supplémentaires ignorées: {extra_features[:5]}", "feature_validation")
            
        return True

    def reset(self):
        """Réinitialiser l'ensemble"""
        self.models = {}
        self.weights = {}
        self.feature_names = []
        self.is_trained = False
        self.error_handler.log_info("Ensemble réinitialisé", "ensemble_reset")
