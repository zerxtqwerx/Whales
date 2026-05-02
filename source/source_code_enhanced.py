"""
source_code_optimized.py
Без SMOTE, но с правильной регуляризацией
"""

import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt, find_peaks, correlate, savgol_filter
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import pickle
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# 1. АУДИО-ОБРАБОТКА (твой код, без изменений)
# ============================================================================

def whale_denoise(y, sr, low_freq=10, high_freq=800):
    try:
        if not np.all(np.isfinite(y)) or len(y) == 0:
            return y
        
        def butter_bandpass(lowcut, highcut, fs, order=4):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a
        
        b, a = butter_bandpass(low_freq, high_freq, sr)
        y_filtered = filtfilt(b, a, y)
        
        if not np.all(np.isfinite(y_filtered)):
            return y
        
        D = librosa.stft(y_filtered, n_fft=2048, hop_length=512)
        magnitude = np.abs(D)
        phase = np.angle(D)
        noise_frames = magnitude[:, :max(1, int(0.2 * sr / 512))]
        noise_profile = np.median(noise_frames, axis=1, keepdims=True)
        noise_profile = np.maximum(noise_profile, 1e-12)
        threshold = 1.5 * noise_profile
        magnitude_clean = np.maximum(magnitude - threshold, 1e-12)
        
        D_clean = magnitude_clean * np.exp(1j * phase)
        y_clean = librosa.istft(D_clean)
        
        return y_clean if np.all(np.isfinite(y_clean)) else y
        
    except Exception:
        return y

def extract_spectrogram_features(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_S = librosa.power_to_db(S, ref=np.max)
    features = {
        'mel_mean': np.mean(log_S, axis=1),
        'mel_std': np.std(log_S, axis=1),
        'mel_skew': skew(log_S, axis=1),
        'mel_kurtosis': kurtosis(log_S, axis=1)
    }
    return features

def extract_single_file_features(audio_file, labels_df, apply_denoise, sample_type='original'):
    y, sr = librosa.load(audio_file, sr=22050, duration=5, mono=True)
    
    if apply_denoise:
        y = whale_denoise(y, sr)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
    mfcc_delta = librosa.feature.delta(mfcc)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    spectral_centroid = librosa.feature.spectral_centroid(S=D, sr=sr)
    spectral_centroid_smooth = savgol_filter(spectral_centroid[0], window_length=11, polyorder=3)
    
    peaks, properties = find_peaks(
        spectral_centroid_smooth,
        height=np.mean(spectral_centroid_smooth) * 1.2,
        distance=sr//100,
        prominence=0.1
    )
    
    if len(peaks) > 1:
        inter_peak_distances = np.diff(peaks)
        bead_regularity = np.std(inter_peak_distances) / (np.mean(inter_peak_distances) + 1e-10)
        bead_mean_distance = np.mean(inter_peak_distances)
        bead_height_variation = np.std(properties['peak_heights'])
    else:
        bead_regularity = 0
        bead_mean_distance = 0
        bead_height_variation = 0
    
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_energy = np.sum(harmonic**2)
    percussive_energy = np.sum(percussive**2)
    harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-10)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=D, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(S=D, sr=sr, roll_percent=0.85)
    spectral_flatness = librosa.feature.spectral_flatness(S=D)
    spectral_contrast = librosa.feature.spectral_contrast(S=D, sr=sr)
    spectral_skewness = skew(D.flatten())
    spectral_kurtosis = kurtosis(D.flatten())
    spectral_crest = np.max(D) / (np.mean(D) + 1e-10)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(S=D)
    autocorr = correlate(y, y)
    autocorr_norm = autocorr / np.max(autocorr)
    periodicity_score = np.sum(autocorr_norm[len(autocorr)//2 + sr//10: len(autocorr)//2 + sr//2]) / (sr//2 - sr//10)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectrogram_features = extract_spectrogram_features(y, sr)
    
    label = None
    if labels_df is not None and 'label' in labels_df.columns:
        filename = audio_file.name
        matching_row = labels_df[labels_df['clip_name'] == filename]
        if not matching_row.empty:
            label = matching_row['label'].iloc[0]
    
    features_data = {
        'filename': audio_file.name,
        'filepath': str(audio_file),
        'sample_type': sample_type,
        'label': label,
        'duration': len(y) / sr,
        'sample_rate': sr,
        'mfcc_mean': np.mean(mfcc, axis=1),
        'mfcc_std': np.std(mfcc, axis=1),
        'mfcc_delta_mean': np.mean(mfcc_delta, axis=1),
        'chroma_stft_mean': np.mean(chroma_stft, axis=1),
        'chroma_stft_std': np.std(chroma_stft, axis=1),
        'chroma_cqt_mean': np.mean(chroma_cqt, axis=1),
        'num_beads': len(peaks),
        'bead_regularity': bead_regularity,
        'bead_mean_distance': bead_mean_distance,
        'bead_height_variation': bead_height_variation,
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_centroid_std': np.std(spectral_centroid),
        'spectral_centroid_range': np.ptp(spectral_centroid),
        'spectral_centroid_skew': skew(spectral_centroid[0]),
        'harmonic_ratio': harmonic_ratio,
        'harmonic_energy': harmonic_energy,
        'percussive_energy': percussive_energy,
        'harmonic_percussive_ratio': harmonic_energy / (percussive_energy + 1e-10),
        'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
        'spectral_bandwidth_std': np.std(spectral_bandwidth),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'spectral_rolloff_std': np.std(spectral_rolloff),
        'spectral_flatness_mean': np.mean(spectral_flatness),
        'spectral_flatness_std': np.std(spectral_flatness),
        'spectral_contrast_mean': np.mean(spectral_contrast, axis=1),
        'spectral_skewness': spectral_skewness,
        'spectral_kurtosis': spectral_kurtosis,
        'spectral_crest': spectral_crest,
        'zcr_mean': np.mean(zero_crossing_rate),
        'zcr_std': np.std(zero_crossing_rate),
        'rms_mean': np.mean(rms),
        'rms_std': np.std(rms),
        'periodicity_score': periodicity_score,
        'tempo': tempo,
        'energy_mean': np.mean(y**2),
        'energy_std': np.std(y**2),
        'mel_mean': spectrogram_features['mel_mean'],
        'mel_std': spectrogram_features['mel_std'],
        'mel_skew': spectrogram_features['mel_skew'],
        'mel_kurtosis': spectrogram_features['mel_kurtosis'],
        'dominant_chroma': np.argmax(np.mean(chroma_stft, axis=1)),
        'chroma_energy_mean': np.mean(np.sum(chroma_stft, axis=0)),
        'chroma_energy_std': np.std(np.sum(chroma_stft, axis=0)),
    }
    
    return features_data

def extract_features(audio_files, labels_df=None, max_files=None, apply_denoise=True):
    features_list = []
    files_to_process = audio_files[:max_files] if max_files else audio_files
    
    for i, audio_file in enumerate(files_to_process):
        if i % 100 == 0:
            print(f"{i}/{len(files_to_process)}")
        
        try:
            features_data = extract_single_file_features(audio_file, labels_df, apply_denoise, 'original')
            features_list.append(features_data)
        except Exception as e:
            continue
    
    print(f"\nУспешно обработано {len(features_list)} из {len(files_to_process)} файлов")
    return pd.DataFrame(features_list)

# ============================================================================
# 2. ПОДГОТОВКА ПРИЗНАКОВ (твой код)
# ============================================================================

def prepare_features_for_training(features_df):
    features_list = []
    
    for _, row in features_df.iterrows():
        try:
            feature_parts = []
            feature_parts.append(row['mfcc_mean'])
            feature_parts.append(row['mfcc_std'])
            feature_parts.append(row['mfcc_delta_mean'])
            feature_parts.append(row['chroma_stft_mean'])
            feature_parts.append(row['chroma_stft_std'])
            feature_parts.append(row['chroma_cqt_mean'])
            
            bead_features = np.array([
                row['num_beads'], row['bead_regularity'], row['bead_mean_distance'],
                row['bead_height_variation'], row['spectral_centroid_mean'],
                row['spectral_centroid_std'], row['spectral_centroid_range'],
                row['spectral_centroid_skew']
            ])
            feature_parts.append(bead_features)
            
            tonal_features = np.array([
                row['harmonic_ratio'], row['harmonic_energy'],
                row['percussive_energy'], row['harmonic_percussive_ratio']
            ])
            feature_parts.append(tonal_features)
            
            if isinstance(row['spectral_contrast_mean'], np.ndarray):
                if row['spectral_contrast_mean'].ndim == 2:
                    spectral_contrast_flat = row['spectral_contrast_mean'].flatten()
                else:
                    spectral_contrast_flat = row['spectral_contrast_mean']
            else:
                spectral_contrast_flat = np.array([row['spectral_contrast_mean']])
            
            spectral_scalars = np.array([
                row['spectral_bandwidth_mean'], row['spectral_bandwidth_std'],
                row['spectral_rolloff_mean'], row['spectral_rolloff_std'],
                row['spectral_flatness_mean'], row['spectral_flatness_std']
            ])
            spectral_features = np.concatenate([spectral_contrast_flat, spectral_scalars])
            feature_parts.append(spectral_features)
            
            stat_features = np.array([
                row['spectral_skewness'], row['spectral_kurtosis'], row['spectral_crest']
            ])
            feature_parts.append(stat_features)
            
            temporal_features = np.array([
                row['zcr_mean'], row['zcr_std'], row['rms_mean'], row['rms_std'],
                row['periodicity_score'], row['tempo']
            ])
            feature_parts.append(temporal_features)
            
            energy_features = np.array([row['energy_mean'], row['energy_std']])
            feature_parts.append(energy_features)
            
            if 'mel_mean' in row and isinstance(row['mel_mean'], np.ndarray):
                feature_parts.append(row['mel_mean'])
                feature_parts.append(row['mel_std'])
                feature_parts.append(row['mel_skew'])
                feature_parts.append(row['mel_kurtosis'])
            
            extra_features = np.array([
                row['dominant_chroma'], row['chroma_energy_mean'], row['chroma_energy_std']
            ])
            feature_parts.append(extra_features)
            
            flattened_parts = []
            for part in feature_parts:
                if isinstance(part, np.ndarray):
                    if part.ndim == 1:
                        flattened_parts.append(part)
                    else:
                        flattened_parts.append(part.flatten())
                else:
                    flattened_parts.append(np.array([part]))
            
            feature_vector = np.concatenate(flattened_parts)
            features_list.append(feature_vector)
            
        except Exception as e:
            continue
    
    if len(features_list) == 0:
        return np.array([])
    
    X = np.array(features_list)
    print(f"Создана матрица признаков: {X.shape}")
    return X

def clean_nan(X, y, name="данные"):
    nan_mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[nan_mask]
    y_clean = y[nan_mask]
    removed = len(X) - len(X_clean)
    if removed > 0:
        print(f"   {name}: удалено {removed} строк с NaN")
    return X_clean, y_clean

# ============================================================================
# 3. ОБУЧЕНИЕ БЕЗ SMOTE, НО С РЕГУЛЯРИЗАЦИЕЙ
# ============================================================================

def train_models_regularized(X_train_scaled, y_train, X_val_scaled, y_val):
    """Обучение с регуляризацией, без SMOTE"""
    
    # Подсчет весов классов (как в твоей статье)
    n_0 = np.sum(y_train == 0)
    n_1 = np.sum(y_train == 1)
    class_weight = {0: 1.0, 1: n_0 / n_1}
    
    print(f"\n📊 Веса классов: 0={class_weight[0]:.2f}, 1={class_weight[1]:.2f}")
    
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=150,           # больше деревьев
            max_depth=5,                # меньше глубина (было 6-7)
            learning_rate=0.03,         # меньше скорость (было 0.05)
            subsample=0.7,              # выборка 70% данных для каждого дерева
            colsample_bytree=0.7,       # выборка 70% признаков
            reg_alpha=0.5,              # L1 регуляризация
            reg_lambda=1.5,             # L2 регуляризация
            scale_pos_weight=n_0 / n_1, # вес класса 1
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=12,               # меньше глубина (было 15)
            min_samples_split=10,       # больше мин. примеров для split
            min_samples_leaf=5,         # больше мин. примеров в листе
            max_features='sqrt',        # меньше признаков на разбиение
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    trained = {}
    results = {}
    
    for name, model in models.items():
        print(f"\nОбучение: {name}")
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        
        acc = accuracy_score(y_val, y_pred)
        f1_weighted = f1_score(y_val, y_pred, average='weighted')
        f1_class1 = f1_score(y_val, y_pred, pos_label=1)
        
        print(f"   Val Accuracy: {acc:.4f}")
        print(f"   Val F1 (weighted): {f1_weighted:.4f}")
        print(f"   Val F1 (киты): {f1_class1:.4f}")
        
        trained[name] = model
        results[name] = {
            'accuracy': acc,
            'f1_weighted': f1_weighted,
            'f1_class1': f1_class1
        }
    
    return trained, results

# ============================================================================
# 4. СОХРАНЕНИЕ
# ============================================================================

def save_all(model, scaler, model_name, val_results, test_results, y_test, y_pred, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'best_model_{timestamp}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(f'scaler_{timestamp}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    metrics = {
        'timestamp': timestamp,
        'best_model': model_name,
        'val_metrics': val_results,
        'test_metrics': test_results
    }
    
    with open(f'metrics_{timestamp}.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Сохраняем classification report
    report = classification_report(y_test, y_pred, target_names=['Не кит', 'Кит'], output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f'report_{timestamp}.csv')
    
    print(f"\n✅ Сохранено: best_model_{timestamp}.pkl, scaler_{timestamp}.pkl, metrics_{timestamp}.json")
    
    return timestamp

# ============================================================================
# 3.1. Функция для поиска оптимального порога
# ============================================================================

def find_optimal_threshold(model, X_val, y_val, metric='f1_class1'):
    """Поиск оптимального порога для максимизации F1 класса 'кит'"""
    
    if not hasattr(model, 'predict_proba'):
        return 0.5, f1_score(y_val, model.predict(X_val), pos_label=1)
    
    probs = model.predict_proba(X_val)[:, 1]
    
    # Перебираем пороги
    thresholds = np.arange(0.3, 0.8, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        pred = (probs >= thresh).astype(int)
        f1 = f1_score(y_val, pred, pos_label=1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    print(f"\n  Оптимальный порог: {best_thresh:.2f} (F1={best_f1:.4f})")
    print(f"  Стандартный порог 0.5: F1={f1_score(y_val, (probs>=0.5).astype(int), pos_label=1):.4f}")
    
    return best_thresh, best_f1


def predict_with_threshold(model, X, threshold=0.5):
    """Предсказание с заданным порогом"""
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
    return model.predict(X)


# ============================================================================
# 3.2. Функция для Stacking ансамбля
# ============================================================================

def train_stacking_ensemble(X_train, y_train, X_val, y_val, X_test, scaler):
    """Обучает стек для XGBoost + Random Forest"""
    
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    print("\n" + "="*60)
    print("🔗 ОБУЧЕНИЕ STACKING АНСАМБЛЯ")
    print("="*60)
    
    # Базовые модели (уже с регуляризацией)
    xgb = XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.5,
        random_state=42, use_label_encoder=False,
        eval_metric='logloss', n_jobs=-1
    )
    
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12,
        min_samples_split=10, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    
    # Мета-модель
    meta = LogisticRegression(C=0.1, class_weight='balanced', random_state=42, max_iter=1000)
    
    # Stacking
    stacking = StackingClassifier(
        estimators=[('xgb', xgb), ('rf', rf)],
        final_estimator=meta,
        cv=3,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    # Обучаем на тренировочных данных (уже масштабированных)
    stacking.fit(X_train, y_train)
    
    # Оценка на валидации
    y_val_pred = stacking.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, pos_label=1)
    val_f1_weighted = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"\n  Stacking на валидации:")
    print(f"    F1 (киты): {val_f1:.4f}")
    print(f"    F1 (weighted): {val_f1_weighted:.4f}")
    
    # Оптимальный порог для стека
    if hasattr(stacking, 'predict_proba'):
        probs_val = stacking.predict_proba(X_val)[:, 1]
        thresholds = np.arange(0.3, 0.8, 0.01)
        best_f1 = 0
        best_thresh = 0.5
        for thresh in thresholds:
            pred = (probs_val >= thresh).astype(int)
            f1 = f1_score(y_val, pred, pos_label=1)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        print(f"\n  Оптимальный порог для стека: {best_thresh:.2f}")
    else:
        best_thresh = 0.5
    
    # Тест
    y_test_pred = predict_with_threshold(stacking, X_test, best_thresh)
    
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
    test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, pos_label=1)
    test_precision = precision_score(y_test, y_test_pred, pos_label=1)
    
    print(f"\n  Stacking на тесте (порог={best_thresh:.2f}):")
    print(f"    F1 (киты): {test_f1:.4f}")
    print(f"    F1 (weighted): {test_f1_weighted:.4f}")
    print(f"    Recall: {test_recall:.4f}")
    print(f"    Precision: {test_precision:.4f}")
    
    return stacking, best_thresh, {
        'val_f1': val_f1,
        'test_f1': test_f1,
        'test_f1_weighted': test_f1_weighted,
        'test_recall': test_recall,
        'test_precision': test_precision
    }


# ============================================================================
# 3.3. Обновлённая функция обучения с threshold tuning
# ============================================================================

def train_models_regularized(X_train_scaled, y_train, X_val_scaled, y_val):
    """Обучение с регуляризацией и поиском оптимального порога"""
    
    n_0 = np.sum(y_train == 0)
    n_1 = np.sum(y_train == 1)
    class_weight = {0: 1.0, 1: n_0 / n_1}
    
    print(f"\n📊 Веса классов: 0={class_weight[0]:.2f}, 1={class_weight[1]:.2f}")
    
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=1.5,
            scale_pos_weight=n_0 / n_1,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', n_jobs=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=12,
            min_samples_split=10, min_samples_leaf=5,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        )
    }
    
    trained = {}
    results = {}
    
    for name, model in models.items():
        print(f"\nОбучение: {name}")
        
        model.fit(X_train_scaled, y_train)
        
        # Стандартный порог 0.5
        y_pred_05 = model.predict(X_val_scaled)
        f1_05 = f1_score(y_val, y_pred_05, pos_label=1)
        f1_weighted_05 = f1_score(y_val, y_pred_05, average='weighted')
        
        # Поиск оптимального порога
        if hasattr(model, 'predict_proba'):
            best_thresh, best_f1 = find_optimal_threshold(model, X_val_scaled, y_val)
            y_pred_opt = predict_with_threshold(model, X_val_scaled, best_thresh)
            f1_opt = f1_score(y_val, y_pred_opt, pos_label=1)
        else:
            best_thresh = 0.5
            best_f1 = f1_05
            f1_opt = f1_05
        
        print(f"\n   Результаты на валидации:")
        print(f"     Порог 0.5:    F1 (киты)={f1_05:.4f}, F1 (weighted)={f1_weighted_05:.4f}")
        print(f"     Порог {best_thresh:.2f}: F1 (киты)={f1_opt:.4f} (+{f1_opt-f1_05:.4f})")
        
        trained[name] = {'model': model, 'threshold': best_thresh}
        results[name] = {
            'accuracy': accuracy_score(y_val, y_pred_05),
            'f1_weighted': f1_weighted_05,
            'f1_class1_05': f1_05,
            'f1_class1_opt': f1_opt,
            'optimal_threshold': best_thresh
        }
    
    return trained, results


# ============================================================================
# 4. ОБНОВЛЁННЫЙ ГЛАВНЫЙ ПАЙПЛАЙН
# ============================================================================

def main_pipeline(train_audio_files, train_df, n=20000, use_stacking=True):
    print("\n" + "="*70)
    print("🐋 ОБУЧЕНИЕ: РЕГУЛЯРИЗАЦИЯ + THRESHOLD TUNING + STACKING")
    print("="*70)
    
    # 1. Извлечение признаков
    print(f"\nИзвлечение признаков из {n} файлов...")
    features_df = extract_features(train_audio_files[:n], labels_df=train_df, max_files=n)
    
    if features_df is None or len(features_df) == 0:
        print("Ошибка")
        return None
    
    print(f"Тренировочные признаки: {features_df.shape}")
    
    # 2. Подготовка
    X = prepare_features_for_training(features_df)
    y = features_df['label'].values
    
    # 3. Разделение
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nРазделение:")
    print(f"   Train: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # 4. Очистка
    X_train, y_train = clean_nan(X_train, y_train, "Train")
    X_val, y_val = clean_nan(X_val, y_val, "Validation")
    X_test, y_test = clean_nan(X_test, y_test, "Test")
    
    # 5. Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Обучение отдельных моделей с threshold tuning
    trained, results = train_models_regularized(X_train_scaled, y_train, X_val_scaled, y_val)
    
    if not trained:
        return None
    
    # 7. Выбор лучшей модели
    best_name = max(results, key=lambda x: results[x]['f1_class1_opt'])
    best_model = trained[best_name]['model']
    best_threshold = results[best_name]['optimal_threshold']
    
    print(f"\n🏆 Лучшая модель: {best_name}")
    print(f"   Оптимальный порог: {best_threshold:.2f}")
    print(f"   Val F1 (киты, opt): {results[best_name]['f1_class1_opt']:.4f}")
    
    # 8. Тест с оптимальным порогом
    y_pred_test = predict_with_threshold(best_model, X_test_scaled, best_threshold)
    
    test_f1 = f1_score(y_test, y_pred_test, pos_label=1)
    test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, pos_label=1)
    test_precision = precision_score(y_test, y_pred_test, pos_label=1)
    
    print(f"\n📊 ТЕСТОВЫЕ РЕЗУЛЬТАТЫ (порог={best_threshold:.2f}):")
    print(f"   F1 (киты): {test_f1:.4f}")
    print(f"   F1 (weighted): {test_f1_weighted:.4f}")
    print(f"   Recall (киты): {test_recall:.4f}")
    print(f"   Precision (киты): {test_precision:.4f}")
    
    # 9. Stacking ансамбль (опционально)
    stacking_results = None
    if use_stacking and len(trained) >= 2:
        stacking, stack_thresh, stack_metrics = train_stacking_ensemble(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, scaler
        )
        
        # Сравнение с лучшей отдельной моделью
        print(f"\n📊 СРАВНЕНИЕ:")
        print(f"   Лучшая отдельная модель ({best_name}): F1={test_f1:.4f}")
        print(f"   Stacking ансамбль:                   F1={stack_metrics['test_f1']:.4f}")
        
        if stack_metrics['test_f1'] > test_f1:
            print(f"   ✅ Stacking лучше на {(stack_metrics['test_f1']-test_f1)*100:.2f}%")
            best_model = stacking
            best_threshold = stack_thresh
            test_f1 = stack_metrics['test_f1']
            test_recall = stack_metrics['test_recall']
            test_precision = stack_metrics['test_precision']
            best_name = "Stacking (XGB+RF)"
        else:
            print(f"   ℹ️ Отдельная модель лучше")
        
        stacking_results = stack_metrics
    
    # 10. Сохранение
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'best_model_{timestamp}.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(f'scaler_{timestamp}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    metrics = {
        'timestamp': timestamp,
        'best_model': best_name,
        'optimal_threshold': best_threshold,
        'val_metrics': results[best_name if best_name != "Stacking (XGB+RF)" else 'XGBoost'],
        'test_metrics': {
            'f1_class1': test_f1,
            'f1_weighted': test_f1_weighted,
            'recall': test_recall,
            'precision': test_precision
        },
        'stacking_used': use_stacking,
        'stacking_results': stacking_results
    }
    
    with open(f'metrics_{timestamp}.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # 11. Финальный отчет
    print("\n" + "="*70)
    print("📊 ФИНАЛЬНЫЙ ОТЧЕТ")
    print("="*70)
    print(f"   Лучшая модель: {best_name}")
    print(f"   Оптимальный порог: {best_threshold:.2f}")
    print(f"   Test F1 (киты): {test_f1:.4f}")
    print(f"   Test Recall (киты): {test_recall:.4f}")
    print(f"   Test Precision (киты): {test_precision:.4f}")
    print(f"\n✅ Сохранено: best_model_{timestamp}.pkl, metrics_{timestamp}.json")
    
    # 12. Матрица ошибок
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Не кит', 'Кит'],
                yticklabels=['Не кит', 'Кит'])
    plt.title(f'Матрица ошибок - {best_name} (порог={best_threshold:.2f})')
    plt.ylabel('Фактические')
    plt.xlabel('Предсказанные')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{timestamp}.png', dpi=150)
    plt.show()
    
    return best_model, scaler, metrics


# ============================================================================
# 5. ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    large_data_path = r'F:\datasets\whale-detection-challenge\whale_data\data'
    train_path = Path(large_data_path) / 'train'
    train_csv_path = Path(large_data_path) / 'train.csv'
    
    train_df = pd.read_csv(train_csv_path)
    train_df = train_df.rename(columns={'clip_name_label': 'clip_name', 'train1.atif,0': 'label'})
    
    train_audio_files = list(train_path.rglob('*.aiff')) + list(train_path.rglob('*.aif'))
    print(f"Найдено файлов: {len(train_audio_files)}")
    print(train_df['label'].value_counts())
    
    # ЗАПУСК С ПАРАМЕТРАМИ
    model, scaler, metrics = main_pipeline(
        train_audio_files, 
        train_df, 
        n=20000,
        use_stacking=True  # True = Stacking, False = только отдельные модели
    )