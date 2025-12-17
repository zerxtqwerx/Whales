# %%
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt, find_peaks, correlate, savgol_filter
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# %%
large_data_path = r'F:\datasets\whale-detection-challenge\whale_data\data'
train_path = Path(large_data_path) / 'train'
train_csv_path = Path(large_data_path) / 'train.csv'
train_df = pd.read_csv(train_csv_path)
train_df = train_df.rename(columns={'clip_name_label': 'filename', 'train1.atif,0': 'label'})
train_audio_files = list(train_path.rglob('*.aiff')) + list(train_path.rglob('*.aif'))

print(f"Размер датасета: {len(train_df)}")
print(f"Распределение меток:\n{train_df['label'].value_counts()}")

# %%
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
        
        if not np.all(np.isfinite(magnitude_clean)):
            return y_filtered
        
        D_clean = magnitude_clean * np.exp(1j * phase)
        y_clean = librosa.istft(D_clean)
        
        if not np.all(np.isfinite(y_clean)):
            return y
            
        return y_clean
        
    except Exception as e:
        return y

# %%
def augment_audio(y, sr):
    augmented = []
    noise = np.random.normal(0, 0.005, len(y))
    augmented.append(y + noise)
    shift = np.random.randint(sr//10)
    augmented.append(np.roll(y, shift))
    
    if len(y) > sr:
        steps = np.random.uniform(-2, 2)
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=steps))
    
    return augmented

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
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
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
        bead_regularity = np.std(inter_peak_distances) / np.mean(inter_peak_distances)
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

def extract_features(audio_files, labels_df=None, max_files=None, apply_denoise=True, augment_data=False):
    features_list = []
    files_to_process = audio_files[:max_files] if max_files else audio_files
    
    for i, audio_file in enumerate(files_to_process):
        if i % 100 == 0:
            print(f"{i}/{len(files_to_process)}")
        
        try:
            y, sr = librosa.load(audio_file, sr=22050, duration=5, mono=True)
            
            if apply_denoise:
                y = whale_denoise(y, sr)
            
            features_data = extract_single_file_features(audio_file, labels_df, apply_denoise, 'original')
            features_list.append(features_data)
            
            if augment_data:
                try:
                    augmented = augment_audio(y, sr)
                    for j, aug_y in enumerate(augmented):
                        temp_file = type('TempFile', (), {'name': f'{audio_file.name}_aug{j}'})()
                        features_aug = extract_single_file_features(temp_file, labels_df, False, f'aug_{j}')
                        features_aug['filename'] = audio_file.name
                        features_aug['filepath'] = str(audio_file)
                        features_list.append(features_aug)
                except Exception as e:
                    print(f"Ошибка аугментации: {e}")
                    
        except Exception as e:
            print(f"Ошибка обработки {audio_file}: {str(e)[:100]}")
            continue
    
    print(f"\nУспешно обработано {len(features_list)} из {len(files_to_process)} файлов")
    
    if augment_data and len(features_list) > 0:
        orig_count = len([f for f in features_list if f['sample_type'] == 'original'])
        aug_count = len(features_list) - orig_count
        print(f"  Оригинальных: {orig_count}")
        print(f"  Аугментированных: {aug_count}")
    
    return pd.DataFrame(features_list)

# %%
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
                row['num_beads'],
                row['bead_regularity'],
                row['bead_mean_distance'],
                row['bead_height_variation'],
                row['spectral_centroid_mean'],
                row['spectral_centroid_std'],
                row['spectral_centroid_range'],
                row['spectral_centroid_skew']
            ])
            feature_parts.append(bead_features)
            
            tonal_features = np.array([
                row['harmonic_ratio'],
                row['harmonic_energy'],
                row['percussive_energy'],
                row['harmonic_percussive_ratio']
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
                row['spectral_bandwidth_mean'],
                row['spectral_bandwidth_std'],
                row['spectral_rolloff_mean'],
                row['spectral_rolloff_std'],
                row['spectral_flatness_mean'],
                row['spectral_flatness_std']
            ])
            spectral_features = np.concatenate([spectral_contrast_flat, spectral_scalars])
            feature_parts.append(spectral_features)
            
            stat_features = np.array([
                row['spectral_skewness'],
                row['spectral_kurtosis'],
                row['spectral_crest']
            ])
            feature_parts.append(stat_features)
            
            temporal_features = np.array([
                row['zcr_mean'],
                row['zcr_std'],
                row['rms_mean'],
                row['rms_std'],
                row['periodicity_score'],
                row['tempo']
            ])
            feature_parts.append(temporal_features)
            
            energy_features = np.array([
                row['energy_mean'],
                row['energy_std']
            ])
            feature_parts.append(energy_features)
            
            if 'mel_mean' in row and isinstance(row['mel_mean'], np.ndarray):
                feature_parts.append(row['mel_mean'])
                feature_parts.append(row['mel_std'])
                feature_parts.append(row['mel_skew'])
                feature_parts.append(row['mel_kurtosis'])
            
            extra_features = np.array([
                row['dominant_chroma'],
                row['chroma_energy_mean'],
                row['chroma_energy_std']
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
            print(f"Ошибка подготовки признаков: {e}")
            print(f"Типы признаков в строке:")
            for col in ['spectral_contrast_mean', 'mel_mean', 'mfcc_mean']:
                if col in row:
                    val = row[col]
                    print(f"  {col}: type={type(val)}, shape={val.shape if hasattr(val, 'shape') else 'N/A'}, ndim={val.ndim if hasattr(val, 'ndim') else 'N/A'}")
            continue
    
    if len(features_list) == 0:
        print("Не удалось подготовить ни одного вектора признаков")
        return np.array([])
    
    X = np.array(features_list)
    print(f"Создана матрица признаков: {X.shape}")
    print(f"Общее количество признаков: {X.shape[1]}")
    return X

# %%
def train_models(X_train_scaled, y_train, X_val_scaled, y_val):
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            solver='lbfgs',
            n_jobs=-1
        ),
        'SVM': SVC( 
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=False,  
            random_state=42,
            cache_size=1000 
        )
    }
    
    trained_models = {}
    val_results = {}
    
    for name, model in models.items():
        print(f"\nОбучение: {name}")
        
        try:
            if name == 'XGBoost':
                scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
                model.set_params(scale_pos_weight=scale_pos_weight)
            
            model.fit(X_train_scaled, y_train)
            y_val_pred = model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')
            
            print(f"   Val Accuracy:  {val_acc:.4f}")
            print(f"   Val F1-score:  {val_f1:.4f}")
            
            trained_models[name] = model
            val_results[name] = {
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_precision': precision_score(y_val, y_val_pred, average='weighted'),
                'val_recall': recall_score(y_val, y_val_pred, average='weighted')
            }
            
        except Exception as e:
            print(f"Ошибка при обучении {name}: {type(e).__name__} - {str(e)[:100]}")
            continue
    
    return trained_models, val_results

def evaluate_results(best_model, X_test_scaled, y_test, best_model_name):
    print(f"Оценка лучшей модели ({best_model_name}) на тестовых данных:")
    y_test_pred = best_model.predict(X_test_scaled)
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    
    print(f"Test Accuracy:   {test_acc:.4f}")
    print(f"Test F1-score:   {test_f1:.4f}")
    print(f"Test Precision:  {test_precision:.4f}")
    print(f"Test Recall:     {test_recall:.4f}")
    print(f"\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))
    
    cm_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title(f'Матрица ошибок - {best_model_name} (Test)')
    plt.ylabel('Фактические')
    plt.xlabel('Предсказанные')
    plt.show()
    
    return {
        'accuracy': test_acc,
        'f1': test_f1,
        'precision': test_precision,
        'recall': test_recall
    }

def clean_nan(X, y, name="данные"):
    nan_mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[nan_mask]
    y_clean = y[nan_mask]
    removed = len(X) - len(X_clean)
    if removed > 0:
        print(f"   {name}: удалено {removed} строк с NaN")
    return X_clean, y_clean

def main_training_pipeline(train_audio_files, train_df, test_audio_files=None, test_df=None, n=100):
    train_features_df = extract_features(train_audio_files, labels_df=train_df, max_files=n)
    
    if train_features_df is None or len(train_features_df) == 0:
        print("Не удалось извлечь признаки из тренировочных данных")
        return None
    
    print(f"Тренировочные признаки: {train_features_df.shape}")
    
    if test_audio_files is not None:
        test_features_df = extract_features(test_audio_files, labels_df=test_df, max_files=50)
        
        if test_features_df is None or len(test_features_df) == 0:
            print("Предупреждение: Не удалось извлечь тестовые признаки")
            test_features_df = None
        else:
            print(f"Тестовые признаки: {test_features_df.shape}")
    else:
        test_features_df = None
        print("Тестовые данные не предоставлены")
    
    X_train_full = prepare_features_for_training(train_features_df)
    y_train_full = train_features_df['label'].values
    
    if X_train_full is None or len(X_train_full) == 0:
        print("Не удалось подготовить тренировочные признаки")
        return None
    
    if test_features_df is None:
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(
            X_train_full, y_train_full,
            test_size=0.15,
            random_state=42,
            stratify=y_train_full
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_temp, y_train_temp,
            test_size=0.176,
            random_state=42,
            stratify=y_train_temp
        )
        print(f"   Train:       {X_train.shape} ({len(y_train)/len(y_train_full)*100:.1f}%)")
        print(f"   Validation:  {X_val.shape} ({len(y_val)/len(y_train_full)*100:.1f}%)")
        print(f"   Test:        {X_test.shape} ({len(y_test)/len(y_train_full)*100:.1f}%)")
    else:
        X_train = X_train_full
        y_train = y_train_full
        X_test = prepare_features_for_training(test_features_df)
        y_test = test_features_df['label'].values
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )
        print(f"   Train:       {X_train.shape}")
        print(f"   Validation:  {X_val.shape}")
        print(f"   Test:        {X_test.shape}")
    
    X_train, y_train = clean_nan(X_train, y_train, "Train")
    X_val, y_val = clean_nan(X_val, y_val, "Validation")
    X_test, y_test = clean_nan(X_test, y_test, "Test")
    
    if len(X_train) < 20 or len(X_val) < 10 or len(X_test) < 10:
        print(f"Ошибка: Слишком мало данных после очистки")
        print(f" Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return None
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    trained_models, val_results = train_models(X_train_scaled, y_train, X_val_scaled, y_val)
    
    if not val_results:
        print("Не удалось обучить ни одну модель")
        return None
    
    best_model_name = max(val_results, key=lambda x: val_results[x]['val_f1'])
    best_model = trained_models[best_model_name]
    
    print(f"Лучшая модель (по валидации): {best_model_name}")
    print(f"Val F1-score: {val_results[best_model_name]['val_f1']:.4f}")
    print(f"Val Accuracy: {val_results[best_model_name]['val_acc']:.4f}")
    
    print(f"\nАнализ результатов валидации:")
    y_val_pred = best_model.predict(X_val_scaled)
    print(classification_report(y_val, y_val_pred))
    
    test_metrics = evaluate_results(best_model, X_test_scaled, y_test, best_model_name)
    
    final_results = {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'val_metrics': val_results[best_model_name],
        'test_metrics': test_metrics,
        'data_info': {
            'train_size': X_train.shape,
            'val_size': X_val.shape,
            'test_size': X_test.shape,
            'n_classes': len(np.unique(y_train))
        }
    }
    
    return final_results

# %%
train_audio_files = list(train_path.rglob('*.aiff')) + list(train_path.rglob('*.aif'))
print(f"Найдено тренировочных файлов: {len(train_audio_files)}")
audio_count = 20000
result = main_training_pipeline(
    train_audio_files=train_audio_files[:audio_count],
    train_df=train_df,
    test_audio_files=None,
    test_df=None,
    n=audio_count
)

# %% 
def get_processed_data():
    """Функция для получения обработанных данных из основной функции"""
    train_audio_files = list(train_path.rglob('*.aiff')) + list(train_path.rglob('*.aif'))
    print(f"Найдено тренировочных файлов: {len(train_audio_files)}")
    
    n = 100  
    result = main_training_pipeline(
        train_audio_files=train_audio_files[:n],
        train_df=train_df,
        test_audio_files=None,
        test_df=None,
        n=n
    )
    return result

def get_feature_extraction_pipeline(audio_files, labels_df, n_samples=1000):
    features_df = extract_features(
        audio_files, 
        labels_df=labels_df, 
        max_files=n_samples,
        apply_denoise=True,
        augment_data=False
    )
    return features_df

def get_data_for_visualization():
    train_audio_files = list(train_path.rglob('*.aiff')) + list(train_path.rglob('*.aif'))
    
    sample_size = 500
    features_df = get_feature_extraction_pipeline(
        train_audio_files[:sample_size], 
        train_df, 
        n_samples=sample_size
    )
    
    if features_df is not None and len(features_df) > 0:
        X = prepare_features_for_training(features_df)
        y = features_df['label'].values
        return X, y, features_df
    else:
        return None, None, None