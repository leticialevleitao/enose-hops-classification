# src/models/time_series_models.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Any, List

# A lógica de importação robusta permanece a mesma
SKTIME_IMPORTS = {}
try:
    from sktime.classification.kernel_based import RocketClassifier
    SKTIME_IMPORTS['RocketClassifier'] = RocketClassifier
except ImportError as e: print(f"AVISO: RocketClassifier não pôde ser importado. Erro: {e}")

try:
    from sktime.classification.interval_based import TimeSeriesForestClassifier
    SKTIME_IMPORTS['TimeSeriesForestClassifier'] = TimeSeriesForestClassifier
except ImportError:
    try:
        from sktime.classification.ensemble import TimeSeriesForestClassifier
        SKTIME_IMPORTS['TimeSeriesForestClassifier'] = TimeSeriesForestClassifier
        print("AVISO: Usando localização antiga para TimeSeriesForestClassifier (ensemble).")
    except ImportError as e:
        print(f"AVISO: TimeSeriesForestClassifier não pôde ser importado de nenhum local conhecido. Erro: {e}")

try:
    from sktime.classification.feature_based import SignatureClassifier
    SKTIME_IMPORTS['SignatureClassifier'] = SignatureClassifier
except ImportError as e:
     print(f"AVISO: SignatureClassifier não pôde ser importado. Erro: {e}")
    
try:
    from sktime.classification.deep_learning import InceptionTimeClassifier
    SKTIME_IMPORTS['InceptionTimeClassifier'] = InceptionTimeClassifier
except (ImportError, OSError) as e:
    print(f"AVISO: InceptionTimeClassifier não está disponível. Erro: {e}")

try:
    from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    SKTIME_IMPORTS['KNeighborsTimeSeriesClassifier'] = KNeighborsTimeSeriesClassifier
except ImportError as e:
    print(f"AVISO: KNeighborsTimeSeriesClassifier não pôde ser importado. Erro: {e}")


def _filter_data_by_sensors(X, all_sensors, selected_sensors, points_per_sensor):
    if not X.size: return X
    if set(selected_sensors) == set(all_sensors): return X
    sensor_to_idx = {name: i for i, name in enumerate(all_sensors)}
    selected_indices = [sensor_to_idx[name] for name in selected_sensors if name in sensor_to_idx]
    reconstructed_signatures = []
    for full_signature in X:
        parts = []
        for idx in selected_indices:
            start, end = idx * points_per_sensor, (idx + 1) * points_per_sensor
            parts.append(full_signature[start:end])
        reconstructed_signatures.append(np.concatenate(parts))
    return np.vstack(reconstructed_signatures)

class TimeSeriesModels:
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self._initialize_models()

    def _initialize_models(self):
        for name, model_class in SKTIME_IMPORTS.items():
            params = {}
            if name != 'KNeighborsTimeSeriesClassifier':
                params['random_state'] = 42
            if name == 'InceptionTimeClassifier':
                params.update({'n_epochs': 20, 'batch_size': 32})
            elif name == 'TimeSeriesForestClassifier':
                params.update({'n_estimators': 100})
            elif name == 'KNeighborsTimeSeriesClassifier':
                params.update({'n_neighbors': 3})
            elif name == 'RocketClassifier':
                 params.update({'num_kernels': 5000})
            try:
                self.models[name] = model_class(**params)
            except Exception as e:
                print(f"AVISO: {name} não pôde ser inicializado: {e}")
                continue
        
    def get_available_models(self) -> list:
        return list(self.models.keys())

    # --- INÍCIO DA CORREÇÃO ---
    # Adicionamos o parâmetro 'use_optimization' para corresponder à chamada da thread
    def train_model(self, model_name: str, data: Dict[str, Any], data_summary: Dict[str, Any],
                    selected_sensors: List[str], selected_classes: List[str],
                    hyperparams: Dict[str, Any], use_optimization: bool = False) -> Dict[str, Any]:
    # --- FIM DA CORREÇÃO ---
        
        if model_name not in self.models or self.models[model_name] is None:
            raise ValueError(f"Modelo {model_name} não disponível.")

        X_train_full, y_train_full = data['X_train'], data['y_train']; X_val_full, y_val_full = data['X_val'], data['y_val']; X_test_full, y_test_full = data['X_test'], data['y_test']
        train_mask = np.isin(y_train_full, selected_classes); X_train, y_train = X_train_full[train_mask], y_train_full[train_mask]
        val_mask = np.isin(y_val_full, selected_classes); X_val, y_val = X_val_full[val_mask], y_val_full[val_mask]
        test_mask = np.isin(y_test_full, selected_classes); X_test, y_test = X_test_full[test_mask], y_test_full[test_mask]
        all_sensors = data_summary['statistics']['Sensores Utilizados']
        points_per_sensor = data_summary['statistics']['Tamanho Final da Assinatura'] // len(all_sensors)
        X_train = _filter_data_by_sensors(X_train, all_sensors, selected_sensors, points_per_sensor)
        X_val = _filter_data_by_sensors(X_val, all_sensors, selected_sensors, points_per_sensor)
        X_test = _filter_data_by_sensors(X_test, all_sensors, selected_sensors, points_per_sensor)
        
        model = self.models[model_name]
        model.set_params(**hyperparams)
        
        # Lógica de reshape específica para o SignatureClassifier
        if model_name == 'SignatureClassifier':
            n_selected_sensors = len(selected_sensors)
            X_train = X_train.reshape(X_train.shape[0], n_selected_sensors, -1)
            if X_val.size > 0: X_val = X_val.reshape(X_val.shape[0], n_selected_sensors, -1)
            if X_test.size > 0: X_test = X_test.reshape(X_test.shape[0], n_selected_sensors, -1)
        # Lógica de reshape para os outros modelos sktime
        elif len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            if X_val.size > 0 and len(X_val.shape) == 2: X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            if X_test.size > 0 and len(X_test.shape) == 2: X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        model.fit(X_train, y_train)
        metrics = self._evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test)
        result = {'model': model, 'model_name': model_name, 'metrics': metrics, 
                  'best_params': hyperparams, 'used_optimization': False, # Otimização não é usada aqui
                  'selected_sensors': selected_sensors, 'selected_classes': selected_classes}
        self.trained_models[model_name] = result
        return result

    def _evaluate(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        metrics = {}
        if X_train.size > 0: metrics['train_accuracy'] = accuracy_score(y_train, model.predict(X_train))
        if X_val.size > 0: metrics['val_accuracy'] = accuracy_score(y_val, model.predict(X_val))
        if X_test.size > 0:
            y_test_pred = model.predict(X_test)
            metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
            metrics['precision'] = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            unique_labels = np.unique(np.concatenate((y_test, y_test_pred)))
            cm = confusion_matrix(y_test, y_test_pred, labels=unique_labels)
            class_counts = cm.sum(axis=1)
            recalls = np.divide(cm.diagonal(), class_counts, out=np.zeros_like(cm.diagonal(), dtype=float), where=class_counts!=0)
            specificities = []
            for i in range(cm.shape[0]):
                tn = np.sum(cm) - np.sum(cm[i,:]) - np.sum(cm[:,i]) + cm[i,i]
                fp = np.sum(cm[:,i]) - cm[i,i]
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specificities.append(specificity)
            gmeans = np.sqrt(np.array(recalls) * np.array(specificities))
            metrics['gmean'] = np.mean(np.nan_to_num(gmeans))
        return metrics