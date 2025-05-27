"""
Module de gestion du cache intelligent
Optimise les performances en stockant les résultats de calculs coûteux
"""

import time
import hashlib
from functools import wraps


class IntelligentCache:
    """Cache intelligent avec TTL et nettoyage automatique"""
    
    def __init__(self):
        self.memory_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}

    def cache_key(self, func_name, args, kwargs):
        """Générer une clé de cache unique"""
        key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key):
        """Récupérer une valeur du cache"""
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry['timestamp'] < entry['ttl']:
                self.cache_stats['hits'] += 1
                return entry['data']
            else:
                del self.memory_cache[key]

        self.cache_stats['misses'] += 1
        return None

    def set(self, key, value, ttl=3600):
        """Stocker une valeur dans le cache"""
        self.memory_cache[key] = {
            'data': value,
            'timestamp': time.time(),
            'ttl': ttl
        }

        # Nettoyage automatique
        if len(self.memory_cache) > 1000:
            self.cleanup_cache()

    def cleanup_cache(self):
        """Nettoyer le cache"""
        current_time = time.time()
        keys_to_remove = [
            key for key, entry in self.memory_cache.items()
            if current_time - entry['timestamp'] > entry['ttl']
        ]
        for key in keys_to_remove:
            del self.memory_cache[key]

    def cache_decorator(self, ttl=3600):
        """Décorateur pour mise en cache automatique"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = self.cache_key(func.__name__, args, kwargs)
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            return wrapper
        return decorator

    def get_stats(self):
        """Obtenir les statistiques du cache"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / max(total_requests, 1) * 100
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'cache_size': len(self.memory_cache)
        }

    def clear(self):
        """Vider le cache"""
        self.memory_cache.clear()
        self.cache_stats = {'hits': 0, 'misses': 0}
