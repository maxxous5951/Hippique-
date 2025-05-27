"""
Module de gestion d'erreurs robuste
Gère les erreurs avec retry automatique et logging avancé
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, Callable


class RobustErrorHandler:
    """Gestionnaire d'erreurs avec retry automatique et logging"""
    
    def __init__(self):
        self.error_log = []
        self.retry_policies = {
            'data_loading': {'max_retries': 3, 'backoff_factor': 2},
            'model_training': {'max_retries': 2, 'backoff_factor': 3},
            'feature_extraction': {'max_retries': 2, 'backoff_factor': 2},
            'prediction': {'max_retries': 1, 'backoff_factor': 1}
        }

        # Configuration du logging
        self._setup_logging()

    def _setup_logging(self):
        """Configuration du système de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('horse_racing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def robust_execute(self, func: Callable, operation_name: str, *args, **kwargs):
        """Exécution robuste avec retry automatique"""
        policy = self.retry_policies.get(operation_name, {'max_retries': 1, 'backoff_factor': 1})

        for attempt in range(policy['max_retries']):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                error_info = {
                    'operation': operation_name,
                    'attempt': attempt + 1,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'args': str(args)[:100],  # Limiter la taille
                    'kwargs': str(kwargs)[:100]
                }

                self.error_log.append(error_info)
                self.logger.error(f"Erreur {operation_name} (tentative {attempt + 1}): {str(e)}")

                # Si c'est la dernière tentative, relancer l'erreur
                if attempt == policy['max_retries'] - 1:
                    self.logger.error(f"Échec définitif pour {operation_name} après {policy['max_retries']} tentatives")
                    raise

                # Attendre avant le retry
                wait_time = policy['backoff_factor'] ** attempt
                self.logger.info(f"Retry {attempt + 1}/{policy['max_retries']} pour {operation_name} dans {wait_time}s")
                time.sleep(wait_time)

    def log_warning(self, message: str, operation: str = "general"):
        """Logger un avertissement"""
        warning_info = {
            'level': 'WARNING',
            'operation': operation,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.error_log.append(warning_info)
        self.logger.warning(f"[{operation}] {message}")

    def log_info(self, message: str, operation: str = "general"):
        """Logger une information"""
        self.logger.info(f"[{operation}] {message}")

    def get_error_summary(self) -> Dict[str, Any]:
        """Obtenir un résumé des erreurs"""
        if not self.error_log:
            return {'total_errors': 0, 'operations': {}}

        # Compter les erreurs par opération
        operations_count = {}
        error_types = {}
        
        for error in self.error_log:
            op = error.get('operation', 'unknown')
            operations_count[op] = operations_count.get(op, 0) + 1
            
            error_type = error.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            'total_errors': len(self.error_log),
            'operations': operations_count,
            'error_types': error_types,
            'recent_errors': self.error_log[-5:] if len(self.error_log) > 5 else self.error_log
        }

    def clear_log(self):
        """Vider le log d'erreurs"""
        self.error_log.clear()
        self.logger.info("Log d'erreurs vidé")

    def add_retry_policy(self, operation_name: str, max_retries: int, backoff_factor: int = 2):
        """Ajouter une nouvelle politique de retry"""
        self.retry_policies[operation_name] = {
            'max_retries': max_retries,
            'backoff_factor': backoff_factor
        }
        self.logger.info(f"Politique de retry ajoutée pour {operation_name}: {max_retries} tentatives")

    def handle_critical_error(self, error: Exception, operation: str, context: str = ""):
        """Gestion des erreurs critiques"""
        critical_info = {
            'level': 'CRITICAL',
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        self.error_log.append(critical_info)
        self.logger.critical(f"ERREUR CRITIQUE [{operation}]: {str(error)} - {context}")
        
        return critical_info

    def is_recoverable_error(self, error: Exception) -> bool:
        """Déterminer si une erreur est récupérable"""
        # Erreurs non récupérables
        non_recoverable = [
            'SystemExit',
            'KeyboardInterrupt', 
            'MemoryError',
            'ImportError'
        ]
        
        return type(error).__name__ not in non_recoverable
