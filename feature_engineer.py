"""
Module d'ingénierie des features avancées
Extraction et transformation des caractéristiques pour l'IA hippique
"""

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from cache_manager import IntelligentCache


class AdvancedFeatureEngineer:
    """Ingénieur de features avancé pour les courses hippiques"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.cache = IntelligentCache()
        self._unknown_detected = {}

    def safe_extract_gains(self, value, key):
        """Extraction sécurisée des gains depuis un dictionnaire"""
        try:
            if isinstance(value, dict):
                return value.get(key, 0)
            elif isinstance(value, str):
                return 0
            else:
                return 0
        except:
            return 0

    def safe_extract_odds(self, value, key):
        """Extraction sécurisée des cotes depuis un dictionnaire"""
        try:
            if isinstance(value, dict):
                result = value.get(key, 50.0 if key == 'rapport' else False if key == 'favoris' else 0.0)
                # Conversion spéciale pour favoris
                if key == 'favoris':
                    return bool(result)
                return float(result) if result is not None else (50.0 if key == 'rapport' else 0.0)
            elif isinstance(value, str):
                return 50.0 if key == 'rapport' else False if key == 'favoris' else 0.0
            else:
                return 50.0 if key == 'rapport' else False if key == 'favoris' else 0.0
        except:
            return 50.0 if key == 'rapport' else False if key == 'favoris' else 0.0

    def analyze_music_advanced(self, music):
        """Analyse sophistiquée de la musique (historique des performances)"""
        if not isinstance(music, str) or len(music) == 0:
            return 0.0

        score = 0.0
        weight = 1.0
        recent_positions = []

        for i, char in enumerate(music[:10]):  # Analyser les 10 dernières courses
            if char.isdigit():
                position = int(char)
                recent_positions.append(position)

                # Score pondéré par la position
                if position == 1:
                    score += 15 * weight
                elif position == 2:
                    score += 10 * weight
                elif position == 3:
                    score += 7 * weight
                elif position <= 5:
                    score += 4 * weight
                elif position <= 8:
                    score += 1 * weight
                else:
                    score -= 1 * weight

                weight *= 0.8  # Décroissance du poids pour les courses plus anciennes

            elif char == 'D':  # Disqualifié
                score -= 5 * weight
                weight *= 0.8
            elif char == '0' or char == 'a':  # Non placé ou abandon
                score -= 2 * weight
                weight *= 0.8

        return score

    def calculate_consistency(self, music):
        """Calculer la régularité des performances"""
        if not isinstance(music, str) or len(music) == 0:
            return 0.0

        positions = []
        for char in music[:8]:
            if char.isdigit():
                positions.append(int(char))

        if len(positions) < 3:
            return 0.0

        # Coefficient de variation inversé (plus c'est bas, plus c'est régulier)
        mean_pos = np.mean(positions)
        std_pos = np.std(positions)

        if mean_pos == 0:
            return 0.0

        cv = std_pos / mean_pos
        consistency = max(0, 1 - cv)  # Normaliser entre 0 et 1

        return consistency

    def calculate_trend(self, music):
        """Calculer la tendance récente (amélioration/détérioration)"""
        if not isinstance(music, str) or len(music) == 0:
            return 0.0

        positions = []
        for char in music[:6]:
            if char.isdigit():
                positions.append(int(char))

        if len(positions) < 3:
            return 0.0

        # Calculer la pente de régression simple
        x = np.arange(len(positions))
        y = np.array(positions)

        # Inverser car position 1 = meilleur
        y_inverted = 10 - y  # Transformer pour que plus haut = meilleur

        if len(x) > 1:
            slope = np.polyfit(x, y_inverted, 1)[0]
            return slope

        return 0.0

    def get_best_recent_position(self, music):
        """Obtenir la meilleure position récente"""
        if not isinstance(music, str) or len(music) == 0:
            return 10

        best_pos = 10
        for char in music[:5]:  # 5 dernières courses
            if char.isdigit():
                pos = int(char)
                best_pos = min(best_pos, pos)

        return best_pos

    def simple_label_encode(self, values, feature_name):
        """Version ultra-simple et rapide de l'encodage"""
        try:
            # Convertir en série et nettoyer
            clean_values = pd.Series(values).fillna('UNKNOWN').astype(str)

            # Créer mapping simple basé sur fréquence
            value_counts = clean_values.value_counts()

            # Top 20 valeurs les plus fréquentes + UNKNOWN
            top_values = list(value_counts.index[:20]) + ['UNKNOWN']

            # Créer mapping numérique simple
            value_to_code = {val: idx for idx, val in enumerate(top_values)}

            # Encoder avec fallback vers 'UNKNOWN' (code max)
            unknown_code = len(top_values) - 1  # Code pour UNKNOWN
            encoded = [value_to_code.get(val, unknown_code) for val in clean_values]

            print(f"       ✅ {feature_name}: {len(set(encoded))} codes uniques")
            return encoded

        except Exception as e:
            print(f"       ❌ Fallback {feature_name}: {str(e)}")
            return [0] * len(values)

    def safe_label_encode(self, values, feature_name):
        """Encodage sécurisé et rapide qui gère les valeurs inconnues"""
        if not hasattr(self, '_unknown_detected'):
            self._unknown_detected = {}

        print(f"     Encodage {feature_name}...")

        # Si trop de valeurs, utiliser la méthode simple
        if len(values) > 1000:
            print(f"       Mode simple (>1000 valeurs)")
            return self.simple_label_encode(values, feature_name)

        # Convertir en série pandas pour performance
        values_series = pd.Series(values).fillna('UNKNOWN').astype(str)

        if feature_name not in self.label_encoders:
            # Créer un nouvel encodeur
            self.label_encoders[feature_name] = LabelEncoder()
            # Ajouter 'UNKNOWN' comme classe par défaut
            unique_values = list(values_series.unique()) + ['UNKNOWN']
            self.label_encoders[feature_name].fit(unique_values)
            print(f"       Nouvel encodeur créé: {len(unique_values)} classes")

        encoder = self.label_encoders[feature_name]

        # Méthode vectorisée plus rapide
        try:
            # Identifier les valeurs connues et inconnues
            known_mask = values_series.isin(encoder.classes_)

            # Traitement vectorisé des valeurs connues
            encoded_values = np.zeros(len(values_series), dtype=int)

            if known_mask.any():
                known_values = values_series[known_mask]
                encoded_values[known_mask] = encoder.transform(known_values)

            # Traitement des valeurs inconnues
            unknown_mask = ~known_mask
            if unknown_mask.any():
                unknown_values = set(values_series[unknown_mask].unique())
                self._unknown_detected[feature_name] = unknown_values

                # Encoder toutes les inconnues comme 'UNKNOWN'
                try:
                    unknown_code = encoder.transform(['UNKNOWN'])[0]
                    encoded_values[unknown_mask] = unknown_code
                    print(f"       {len(unknown_values)} nouvelles valeurs → 'UNKNOWN'")
                except:
                    encoded_values[unknown_mask] = 0
                    print(f"       {len(unknown_values)} nouvelles valeurs → 0")

            print(f"       ✅ {feature_name} encodé: {len(encoded_values)} valeurs")
            return encoded_values.tolist()

        except Exception as e:
            print(f"       ❌ Erreur encodage {feature_name}: {str(e)}")
            print(f"       Utilisation méthode simple...")
            return self.simple_label_encode(values, feature_name)

    def get_unknown_values_report(self):
        """Générer un rapport des valeurs inconnues détectées"""
        if not hasattr(self, '_unknown_detected') or not self._unknown_detected:
            return None

        report = "🔍 NOUVELLES VALEURS DÉTECTÉES\n"
        report += "=" * 40 + "\n"

        for feature, unknown_set in self._unknown_detected.items():
            if unknown_set:
                report += f"\n📊 {feature.upper()}:\n"
                for value in list(unknown_set)[:5]:  # Limiter à 5
                    report += f"  • {value}\n"
                if len(unknown_set) > 5:
                    report += f"  • ... et {len(unknown_set) - 5} autres\n"

        report += f"\n💡 IMPACT:\n"
        report += f"• Ces valeurs sont traitées comme 'INCONNUES'\n"
        report += f"• Précision légèrement réduite pour ces participants\n"
        report += f"• Réentraîner l'IA avec plus de données améliorerait les prédictions\n"

        return report

    def extract_comprehensive_features(self, df):
        """Extraction complète des features basée sur les vraies données JSON"""
        # Utiliser le cache si disponible
        cache_key = f"features_{hash(str(df.shape))}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        print("🔧 Extraction avancée des features...")

        try:
            features_df = df.copy()

            # ===== FEATURES DE BASE =====
            print("   • Features de base...")
            numeric_cols = ['age', 'nombreCourses', 'nombreVictoires', 'nombrePlaces', 
                           'nombrePlacesSecond', 'nombrePlacesTroisieme']
            
            for col in numeric_cols:
                if col in features_df.columns:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

            # ===== RATIOS DE PERFORMANCE AVANCÉS =====
            print("   • Ratios de performance...")
            features_df['win_rate'] = features_df['nombreVictoires'] / np.maximum(features_df['nombreCourses'], 1)
            features_df['place_rate'] = features_df['nombrePlaces'] / np.maximum(features_df['nombreCourses'], 1)
            features_df['place2_rate'] = features_df['nombrePlacesSecond'] / np.maximum(features_df['nombreCourses'], 1)
            features_df['place3_rate'] = features_df['nombrePlacesTroisieme'] / np.maximum(features_df['nombreCourses'], 1)

            # Ratios combinés
            features_df['top3_rate'] = (features_df['nombreVictoires'] + 
                                       features_df['nombrePlacesSecond'] +
                                       features_df['nombrePlacesTroisieme']) / np.maximum(features_df['nombreCourses'], 1)

            # ===== ANALYSE MUSIQUE SOPHISTIQUÉE =====
            print("   • Analyse musique...")
            if 'musique' in features_df.columns:
                features_df['recent_form_score'] = features_df['musique'].apply(self.analyze_music_advanced)
                features_df['consistency_score'] = features_df['musique'].apply(self.calculate_consistency)
                features_df['trend_score'] = features_df['musique'].apply(self.calculate_trend)
                features_df['best_recent_position'] = features_df['musique'].apply(self.get_best_recent_position)
            else:
                # Valeurs par défaut si musique manquante
                features_df['recent_form_score'] = 0
                features_df['consistency_score'] = 0
                features_df['trend_score'] = 0
                features_df['best_recent_position'] = 10

            # ===== FEATURES FINANCIÈRES AVANCÉES =====
            print("   • Features financières...")
            if 'gainsParticipant' in features_df.columns:
                features_df['gains_carriere'] = features_df['gainsParticipant'].apply(self.safe_extract_gains, args=('gainsCarriere',))
                features_df['gains_victoires'] = features_df['gainsParticipant'].apply(self.safe_extract_gains, args=('gainsVictoires',))
                features_df['gains_place'] = features_df['gainsParticipant'].apply(self.safe_extract_gains, args=('gainsPlace',))
                features_df['gains_annee_courante'] = features_df['gainsParticipant'].apply(self.safe_extract_gains, args=('gainsAnneeEnCours',))
                features_df['gains_annee_precedente'] = features_df['gainsParticipant'].apply(self.safe_extract_gains, args=('gainsAnneePrecedente',))
            else:
                # Valeurs par défaut
                for col in ['gains_carriere', 'gains_victoires', 'gains_place', 'gains_annee_courante', 'gains_annee_precedente']:
                    features_df[col] = 0

            # Ratios financiers
            features_df['avg_gain_per_race'] = features_df['gains_carriere'] / np.maximum(features_df['nombreCourses'], 1)
            features_df['win_gain_ratio'] = features_df['gains_victoires'] / np.maximum(features_df['gains_carriere'], 1)
            features_df['recent_earning_trend'] = np.where(
                features_df['gains_annee_precedente'] > 0,
                features_df['gains_annee_courante'] / features_df['gains_annee_precedente'],
                0
            )

            # ===== FEATURES DE MARCHÉ AVANCÉES =====
            print("   • Features de marché...")
            if 'dernierRapportDirect' in features_df.columns:
                features_df['direct_odds'] = features_df['dernierRapportDirect'].apply(self.safe_extract_odds, args=('rapport',))
                features_df['is_favorite'] = features_df['dernierRapportDirect'].apply(self.safe_extract_odds, args=('favoris',)).astype(int)
            else:
                features_df['direct_odds'] = 50.0
                features_df['is_favorite'] = 0

            if 'dernierRapportReference' in features_df.columns:
                features_df['reference_odds'] = features_df['dernierRapportReference'].apply(self.safe_extract_odds, args=('rapport',))
                features_df['odds_trend'] = features_df['dernierRapportReference'].apply(self.safe_extract_odds, args=('nombreIndicateurTendance',))
            else:
                features_df['reference_odds'] = 50.0
                features_df['odds_trend'] = 0

            # Tendances de cotes
            features_df['odds_movement'] = features_df['direct_odds'] - features_df['reference_odds']
            features_df['odds_volatility'] = np.abs(features_df['odds_movement'])

            # ===== FEATURES COMPÉTITIVES =====
            print("   • Features compétitives...")
            if 'race_file' in features_df.columns:
                # Calculer les features par course
                race_groups = features_df.groupby('race_file')

                # Force du champ de concurrents
                features_df['field_avg_winrate'] = race_groups['win_rate'].transform('mean')
                features_df['field_strength'] = race_groups['avg_gain_per_race'].transform('mean')
                features_df['relative_experience'] = features_df['nombreCourses'] / race_groups['nombreCourses'].transform('mean')

                # Position relative dans le champ
                features_df['winrate_rank'] = race_groups['win_rate'].rank(ascending=False, method='min')
                features_df['earnings_rank'] = race_groups['avg_gain_per_race'].rank(ascending=False, method='min')
                features_df['odds_rank'] = race_groups['direct_odds'].rank(ascending=True, method='min')
            else:
                # Valeurs par défaut
                for col in ['field_avg_winrate', 'field_strength', 'relative_experience', 
                           'winrate_rank', 'earnings_rank', 'odds_rank']:
                    features_df[col] = 0

            # ===== FEATURES CATÉGORIELLES ENCODÉES =====
            print("   • Encodage catégoriel...")
            start_time = time.time()

            categorical_features = ['sexe', 'race', 'driver', 'entraineur', 'proprietaire']
            total_rows = len(features_df)
            use_simple_mode = total_rows > 500

            if use_simple_mode:
                print(f"     Mode simple activé ({total_rows} lignes)")

            for i, feature in enumerate(categorical_features):
                feature_start = time.time()

                if feature in features_df.columns:
                    print(f"     ({i+1}/{len(categorical_features)}) Traitement {feature}...")

                    try:
                        if use_simple_mode:
                            encoded_values = self.simple_label_encode(features_df[feature], feature)
                        else:
                            encoded_values = self.safe_label_encode(features_df[feature], feature)

                        features_df[f'{feature}_encoded'] = encoded_values

                        elapsed = time.time() - feature_start
                        print(f"     ✅ {feature} terminé ({elapsed:.1f}s)")

                        if elapsed > 10:
                            use_simple_mode = True

                    except Exception as e:
                        print(f"     ❌ Erreur {feature}: {str(e)}")
                        features_df[f'{feature}_encoded'] = 0
                else:
                    features_df[f'{feature}_encoded'] = 0

                # Timeout global (2 minutes max)
                total_elapsed = time.time() - start_time
                if total_elapsed > 120:
                    print(f"     ⏰ TIMEOUT encodage - Reste des features → 0")
                    for remaining_feature in categorical_features[i+1:]:
                        features_df[f'{remaining_feature}_encoded'] = 0
                    break

            # ===== FEATURES D'INTERACTION =====
            print("   • Features d'interaction...")
            features_df['age_experience'] = features_df['age'] * features_df['nombreCourses']
            features_df['winrate_odds'] = features_df['win_rate'] * features_df['direct_odds']
            features_df['form_earnings'] = features_df['recent_form_score'] * features_df['avg_gain_per_race']
            features_df['consistency_winrate'] = features_df['consistency_score'] * features_df['win_rate']

            # ===== FEATURES TEMPORELLES =====
            print("   • Features temporelles...")
            if 'race_date' in features_df.columns:
                features_df['day_of_week'] = features_df['race_date'].dt.dayofweek
                features_df['month'] = features_df['race_date'].dt.month
                features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)

            # ===== VARIABLES CIBLES =====
            print("   • Variables cibles...")
            if 'ordreArrivee' in features_df.columns:
                features_df['final_position'] = pd.to_numeric(features_df['ordreArrivee'], errors='coerce')
            else:
                features_df['final_position'] = 10

            features_df['final_position'] = features_df['final_position'].fillna(10)

            # Gérer les statuts spéciaux
            if 'statut' in features_df.columns:
                features_df.loc[features_df['statut'] == 'NON_PARTANT', 'final_position'] = 99

            if 'incident' in features_df.columns:
                incident_series = features_df['incident'].fillna('').astype(str)
                disqualified_mask = incident_series.str.contains('DISQUALIFIE', na=False)
                features_df.loc[disqualified_mask, 'final_position'] = 98

            features_df['won_race'] = (features_df['final_position'] == 1).astype(int)
            features_df['top3_finish'] = (features_df['final_position'] <= 3).astype(int)
            features_df['top5_finish'] = (features_df['final_position'] <= 5).astype(int)

            # ===== NETTOYAGE FINAL =====
            print("   • Nettoyage final...")
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(0)

            print(f"✅ Features extraites: {features_df.shape[1]} colonnes, {features_df.shape[0]} lignes")

            # Sauvegarder dans le cache
            self.cache.set(cache_key, features_df, ttl=1800)

            return features_df

        except Exception as e:
            print(f"❌ Erreur dans extract_comprehensive_features: {str(e)}")
            raise
