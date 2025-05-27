"""
Module d'ingénierie des features avancées avec différenciation Galop/Trot
Extraction et transformation des caractéristiques spécialisées par allure
"""

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from cache_manager import IntelligentCache


class EnhancedFeatureEngineer:
    """Ingénieur de features avancé avec spécialisation Galop/Trot"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {'GALOP': {}, 'TROT': {}}
        self.cache = IntelligentCache()
        self._unknown_detected = {}
        
        # Features spécifiques par allure
        self.galop_specific_features = [
            'handicapPoids', 'oeilleres_encoded', 'distanceChevalPrecedent',
            'jumentPleine', 'supplement'
        ]
        
        self.trot_specific_features = [
            'handicapDistance', 'tempsObtenu', 'reductionKilometrique',
            'deferre_encoded', 'avisEntraineur_encoded'
        ]

    def detect_race_type(self, df):
        """Détecter le type de course (GALOP/TROT) et séparer les données"""
        if 'allure' not in df.columns:
            print("⚠️ Colonne 'allure' manquante - traitement uniforme")
            return {'MIXED': df}
        
        race_types = {}
        unique_allures = df['allure'].unique()
        
        for allure in unique_allures:
            if allure in ['GALOP', 'TROT']:
                subset = df[df['allure'] == allure].copy()
                if len(subset) > 0:
                    race_types[allure] = subset
                    print(f"📊 {allure}: {len(subset)} chevaux détectés")
        
        if not race_types:
            print("⚠️ Aucune allure standard détectée - traitement uniforme")
            race_types['MIXED'] = df
            
        return race_types

    def extract_galop_specific_features(self, df):
        """Extraire les features spécifiques au galop"""
        print("     🏇 Features spécifiques GALOP...")
        
        # Poids et handicaps
        if 'handicapPoids' in df.columns:
            df['handicapPoids'] = pd.to_numeric(df['handicapPoids'], errors='coerce').fillna(55)
            df['poids_relatif'] = df['handicapPoids'] / df['handicapPoids'].mean()
        else:
            df['handicapPoids'] = 55
            df['poids_relatif'] = 1.0
        
        # Oeilières (impact sur la concentration)
        if 'oeilleres' in df.columns:
            df['oeilleres_encoded'] = self.safe_label_encode(df['oeilleres'], 'oeilleres', 'GALOP')
            df['has_oeilleres'] = (df['oeilleres'] != 'SANS_OEILLERES').astype(int)
        else:
            df['oeilleres_encoded'] = 0
            df['has_oeilleres'] = 0
        
        # Distance avec cheval précédent (finish)
        if 'distanceChevalPrecedent' in df.columns:
            df['distance_precedent_code'] = df['distanceChevalPrecedent'].apply(
                lambda x: x.get('code', 0) if isinstance(x, dict) else 0
            )
        else:
            df['distance_precedent_code'] = 0
        
        # Jument pleine (impact sur performance)
        if 'jumentPleine' in df.columns:
            df['jument_pleine'] = df['jumentPleine'].astype(int)
        else:
            df['jument_pleine'] = 0
        
        # Supplément (engagement supplémentaire)
        if 'supplement' in df.columns:
            df['supplement'] = pd.to_numeric(df['supplement'], errors='coerce').fillna(0)
        else:
            df['supplement'] = 0
        
        return df

    def extract_trot_specific_features(self, df):
        """Extraire les features spécifiques au trot"""
        print("     🏇 Features spécifiques TROT...")
        
        # Distance handicap (placement départ)
        if 'handicapDistance' in df.columns:
            df['handicapDistance'] = pd.to_numeric(df['handicapDistance'], errors='coerce').fillna(2100)
            df['handicap_avantage'] = (2100 - df['handicapDistance']) / 100  # Avantage en hectomètres
        else:
            df['handicapDistance'] = 2100
            df['handicap_avantage'] = 0
        
        # Temps obtenu (performance chronométrée)
        if 'tempsObtenu' in df.columns:
            df['tempsObtenu'] = pd.to_numeric(df['tempsObtenu'], errors='coerce')
            df['temps_valide'] = (~df['tempsObtenu'].isna()).astype(int)
            
            # Conversion en vitesse (km/h approximative)
            df['vitesse_kmh'] = np.where(
                df['tempsObtenu'] > 0,
                (df['handicapDistance'] / df['tempsObtenu']) * 3600 / 1000,
                0
            )
        else:
            df['tempsObtenu'] = 0
            df['temps_valide'] = 0
            df['vitesse_kmh'] = 0
        
        # Réduction kilométrique (performance relative)
        if 'reductionKilometrique' in df.columns:
            df['reductionKilometrique'] = pd.to_numeric(df['reductionKilometrique'], errors='coerce').fillna(70000)
            df['reduction_performance'] = 72000 - df['reductionKilometrique']  # Plus c'est bas, mieux c'est
        else:
            df['reductionKilometrique'] = 70000
            df['reduction_performance'] = 2000
        
        # Ferrage (technique de course)
        if 'deferre' in df.columns:
            df['deferre_encoded'] = self.safe_label_encode(df['deferre'], 'deferre', 'TROT')
            df['deferre_posterior'] = df['deferre'].str.contains('POSTERIEURS', na=False).astype(int)
            df['deferre_anterior'] = df['deferre'].str.contains('ANTERIEURS', na=False).astype(int)
        else:
            df['deferre_encoded'] = 0
            df['deferre_posterior'] = 0
            df['deferre_anterior'] = 0
        
        # Avis entraîneur (spécifique au trot)
        if 'avisEntraineur' in df.columns:
            df['avisEntraineur_encoded'] = self.safe_label_encode(df['avisEntraineur'], 'avisEntraineur', 'TROT')
            df['avis_positif'] = (df['avisEntraineur'] == 'POSITIF').astype(int)
        else:
            df['avisEntraineur_encoded'] = 0
            df['avis_positif'] = 0
        
        return df

    def calculate_trot_specific_metrics(self, df, race_groups):
        """Calculer des métriques spécifiques au trot"""
        if 'tempsObtenu' in df.columns:
            # Performance relative dans la course
            df['temps_rank'] = race_groups['tempsObtenu'].rank(method='min')
            df['vitesse_rank'] = race_groups['vitesse_kmh'].rank(ascending=False, method='min')
            
            # Écart au meilleur temps de la course
            df['ecart_meilleur_temps'] = df['tempsObtenu'] - race_groups['tempsObtenu'].transform('min')
        
        if 'reductionKilometrique' in df.columns:
            # Performance relative réduction kilométrique
            df['reduction_rank'] = race_groups['reductionKilometrique'].rank(method='min')
            df['ecart_meilleure_reduction'] = df['reductionKilometrique'] - race_groups['reductionKilometrique'].transform('min')
        
        return df

    def calculate_galop_specific_metrics(self, df, race_groups):
        """Calculer des métriques spécifiques au galop"""
        if 'handicapPoids' in df.columns:
            # Performance relative poids
            df['poids_rank'] = race_groups['handicapPoids'].rank(ascending=False, method='min')  # Plus lourd = plus dur
            df['poids_moyenne_course'] = race_groups['handicapPoids'].transform('mean')
            df['ecart_poids_moyen'] = df['handicapPoids'] - df['poids_moyenne_course']
        
        if 'distance_precedent_code' in df.columns:
            # Analyse des écarts à l'arrivée (finish quality)
            df['finish_quality'] = np.where(
                df['distance_precedent_code'] <= 5,  # Codes faibles = petits écarts
                1,  # Bon finish
                0   # Finish moins bon
            )
        
        return df

    def safe_label_encode(self, values, feature_name, race_type):
        """Encodage sécurisé spécialisé par type de course"""
        if not hasattr(self, '_unknown_detected'):
            self._unknown_detected = {}

        encoder_key = f"{race_type}_{feature_name}"
        print(f"       Encodage {encoder_key}...")

        # Convertir en série pandas
        values_series = pd.Series(values).fillna('UNKNOWN').astype(str)

        if encoder_key not in self.label_encoders[race_type]:
            # Créer encodeur spécialisé pour ce type de course
            self.label_encoders[race_type][feature_name] = LabelEncoder()
            unique_values = list(values_series.unique()) + ['UNKNOWN']
            self.label_encoders[race_type][feature_name].fit(unique_values)
            print(f"         Nouvel encodeur {race_type} créé: {len(unique_values)} classes")

        encoder = self.label_encoders[race_type][feature_name]

        try:
            # Traitement vectorisé
            known_mask = values_series.isin(encoder.classes_)
            encoded_values = np.zeros(len(values_series), dtype=int)

            if known_mask.any():
                encoded_values[known_mask] = encoder.transform(values_series[known_mask])

            # Gestion des valeurs inconnues
            unknown_mask = ~known_mask
            if unknown_mask.any():
                unknown_values = set(values_series[unknown_mask].unique())
                self._unknown_detected[encoder_key] = unknown_values

                try:
                    unknown_code = encoder.transform(['UNKNOWN'])[0]
                    encoded_values[unknown_mask] = unknown_code
                except:
                    encoded_values[unknown_mask] = 0

            return encoded_values.tolist()

        except Exception as e:
            print(f"         ❌ Erreur encodage {encoder_key}: {str(e)}")
            return [0] * len(values)

    def extract_comprehensive_features_by_type(self, df):
        """Extraction complète des features avec spécialisation par type de course"""
        print("🔧 Extraction avancée des features avec spécialisation Galop/Trot...")

        # Séparer par type de course
        race_types_data = self.detect_race_type(df)
        all_processed_data = []

        for race_type, race_df in race_types_data.items():
            print(f"\n🎯 Traitement {race_type} ({len(race_df)} chevaux)...")
            
            # Features communes de base
            processed_df = self.extract_base_features(race_df.copy())
            
            # Features spécialisées
            if race_type == 'GALOP':
                processed_df = self.extract_galop_specific_features(processed_df)
            elif race_type == 'TROT':
                processed_df = self.extract_trot_specific_features(processed_df)
            
            # Features communes avancées
            processed_df = self.extract_common_advanced_features(processed_df, race_type)
            
            # Métriques compétitives spécialisées
            if 'race_file' in processed_df.columns:
                race_groups = processed_df.groupby('race_file')
                
                if race_type == 'GALOP':
                    processed_df = self.calculate_galop_specific_metrics(processed_df, race_groups)
                elif race_type == 'TROT':
                    processed_df = self.calculate_trot_specific_metrics(processed_df, race_groups)
                
                # Métriques communes
                processed_df = self.calculate_common_competitive_features(processed_df, race_groups)
            
            # Variables cibles et nettoyage
            processed_df = self.create_target_variables(processed_df)
            processed_df = processed_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            all_processed_data.append(processed_df)

        # Combiner toutes les données traitées
        final_df = pd.concat(all_processed_data, ignore_index=True)
        
        print(f"✅ Features extraites: {final_df.shape[1]} colonnes, {final_df.shape[0]} lignes")
        print(f"📊 Répartition: {final_df['allure'].value_counts().to_dict()}")
        
        return final_df

    def extract_base_features(self, df):
        """Extraire les features de base communes"""
        print("     • Features de base communes...")
        
        # Features numériques de base
        numeric_cols = ['age', 'nombreCourses', 'nombreVictoires', 'nombrePlaces', 
                       'nombrePlacesSecond', 'nombrePlacesTroisieme']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Ratios de performance
        df['win_rate'] = df['nombreVictoires'] / np.maximum(df['nombreCourses'], 1)
        df['place_rate'] = df['nombrePlaces'] / np.maximum(df['nombreCourses'], 1)
        df['top3_rate'] = (df['nombreVictoires'] + 
                          df['nombrePlacesSecond'] +
                          df['nombrePlacesTroisieme']) / np.maximum(df['nombreCourses'], 1)

        # Analyse musique
        if 'musique' in df.columns:
            df['recent_form_score'] = df['musique'].apply(self.analyze_music_advanced)
            df['consistency_score'] = df['musique'].apply(self.calculate_consistency)
        else:
            df['recent_form_score'] = 0
            df['consistency_score'] = 0

        return df

    def extract_common_advanced_features(self, df, race_type):
        """Extraire les features avancées communes"""
        print("     • Features avancées communes...")
        
        # Features financières
        if 'gainsParticipant' in df.columns:
            df['gains_carriere'] = df['gainsParticipant'].apply(self.safe_extract_gains, args=('gainsCarriere',))
            df['avg_gain_per_race'] = df['gains_carriere'] / np.maximum(df['nombreCourses'], 1)
        else:
            df['gains_carriere'] = 0
            df['avg_gain_per_race'] = 0

        # Features de marché
        if 'dernierRapportDirect' in df.columns:
            df['direct_odds'] = df['dernierRapportDirect'].apply(self.safe_extract_odds, args=('rapport',))
            df['is_favorite'] = df['dernierRapportDirect'].apply(self.safe_extract_odds, args=('favoris',)).astype(int)
        else:
            df['direct_odds'] = 50.0
            df['is_favorite'] = 0

        # Features catégorielles avec encodage spécialisé
        categorical_features = ['sexe', 'race', 'driver', 'entraineur', 'proprietaire']
        
        for feature in categorical_features:
            if feature in df.columns:
                df[f'{feature}_encoded'] = self.safe_label_encode(df[feature], feature, race_type)
            else:
                df[f'{feature}_encoded'] = 0

        return df

    def calculate_common_competitive_features(self, df, race_groups):
        """Calculer les features compétitives communes"""
        df['field_avg_winrate'] = race_groups['win_rate'].transform('mean')
        df['winrate_rank'] = race_groups['win_rate'].rank(ascending=False, method='min')
        df['odds_rank'] = race_groups['direct_odds'].rank(ascending=True, method='min')
        
        return df

    def create_target_variables(self, df):
        """Créer les variables cibles"""
        if 'ordreArrivee' in df.columns:
            df['final_position'] = pd.to_numeric(df['ordreArrivee'], errors='coerce').fillna(10)
        else:
            df['final_position'] = 10

        df['won_race'] = (df['final_position'] == 1).astype(int)
        df['top3_finish'] = (df['final_position'] <= 3).astype(int)
        
        return df

    # Méthodes utilitaires existantes (à conserver)
    def safe_extract_gains(self, value, key):
        try:
            if isinstance(value, dict):
                return value.get(key, 0)
            return 0
        except:
            return 0

    def safe_extract_odds(self, value, key):
        try:
            if isinstance(value, dict):
                result = value.get(key, 50.0 if key == 'rapport' else False if key == 'favoris' else 0.0)
                if key == 'favoris':
                    return bool(result)
                return float(result) if result is not None else (50.0 if key == 'rapport' else 0.0)
            return 50.0 if key == 'rapport' else False if key == 'favoris' else 0.0
        except:
            return 50.0 if key == 'rapport' else False if key == 'favoris' else 0.0

    def analyze_music_advanced(self, music):
        if not isinstance(music, str) or len(music) == 0:
            return 0.0

        score = 0.0
        weight = 1.0

        for char in music[:10]:
            if char.isdigit():
                position = int(char)
                if position == 1:
                    score += 15 * weight
                elif position == 2:
                    score += 10 * weight
                elif position == 3:
                    score += 7 * weight
                elif position <= 5:
                    score += 4 * weight
                weight *= 0.8
            elif char == 'D':
                score -= 5 * weight
                weight *= 0.8

        return score

    def calculate_consistency(self, music):
        if not isinstance(music, str) or len(music) == 0:
            return 0.0

        positions = []
        for char in music[:8]:
            if char.isdigit():
                positions.append(int(char))

        if len(positions) < 3:
            return 0.0

        mean_pos = np.mean(positions)
        std_pos = np.std(positions)

        if mean_pos == 0:
            return 0.0

        cv = std_pos / mean_pos
        consistency = max(0, 1 - cv)

        return consistency

    def get_feature_list_by_type(self, race_type):
        """Obtenir la liste des features spécifiques à un type de course"""
        base_features = [
            'age', 'nombreCourses', 'win_rate', 'place_rate', 'top3_rate',
            'recent_form_score', 'consistency_score', 'gains_carriere', 
            'avg_gain_per_race', 'direct_odds', 'is_favorite',
            'field_avg_winrate', 'winrate_rank', 'odds_rank'
        ]
        
        categorical_features = [f'{cat}_encoded' for cat in 
                              ['sexe', 'race', 'driver', 'entraineur', 'proprietaire']]
        
        if race_type == 'GALOP':
            specific_features = [
                'handicapPoids', 'poids_relatif', 'oeilleres_encoded', 'has_oeilleres',
                'distance_precedent_code', 'jument_pleine', 'supplement',
                'poids_rank', 'finish_quality'
            ]
        elif race_type == 'TROT':
            specific_features = [
                'handicapDistance', 'handicap_avantage', 'tempsObtenu', 'temps_valide',
                'vitesse_kmh', 'reductionKilometrique', 'reduction_performance',
                'deferre_encoded', 'deferre_posterior', 'deferre_anterior',
                'avisEntraineur_encoded', 'avis_positif',
                'temps_rank', 'vitesse_rank', 'reduction_rank'
            ]
        else:
            specific_features = []
        
        return base_features + categorical_features + specific_features

    def get_models_summary_by_type(self):
        """Résumé des modèles par type de course"""
        summary = {
            'GALOP': {
                'specific_features': len(self.galop_specific_features),
                'encoders': len(self.label_encoders.get('GALOP', {}))
            },
            'TROT': {
                'specific_features': len(self.trot_specific_features),
                'encoders': len(self.label_encoders.get('TROT', {}))
            }
        }
        return summary