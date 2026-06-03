# src/models/classical_models.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("AVISO: scikit-optimize não está instalado. A otimização de hiperparâmetros não estará disponível.")


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

class ClassicalModels:
    def __init__(self):
        self.models = {
            # Modelos Clássicos Existentes
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'k-NN': KNeighborsClassifier(),
            'MLP': MLPClassifier(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),

            # ✅ NOVOS: Métodos Lineares (FASE 2 - V12)
            'LDA': LinearDiscriminantAnalysis(),
            'QDA': QuadraticDiscriminantAnalysis(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, multi_class='multinomial'),
            'Naive Bayes': GaussianNB()
        }
        # --- ALTERAÇÃO: Adicionados espaços de busca para MLP e Decision Tree ---
        self.search_spaces = {
            'Random Forest': {
                'n_estimators': Integer(50, 500), 'max_depth': Integer(5, 50),
                'min_samples_split': Integer(2, 10), 'min_samples_leaf': Integer(1, 5)
            },
            'SVM': {
                'C': Real(1e-6, 1e+6, prior='log-uniform'), 'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                'degree': Integer(1, 8), 'kernel': Categorical(['linear', 'poly', 'rbf']),
            },
            'k-NN': {
                'n_neighbors': Integer(1, 50), # Este limite superior será ajustado dinamicamente
                'weights': Categorical(['uniform', 'distance']), 'p': Integer(1, 2)
            },
            'MLP': {
                'hidden_layer_sizes': Categorical([(50,50), (100,), (100,50)]),
                'activation': Categorical(['relu', 'tanh']),
                'alpha': Real(1e-5, 1e-1, prior='log-uniform'),
            },
            'Decision Tree': {
                'max_depth': Integer(2, 50),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
            },
            # ✅ NOVOS: Espaços de busca para métodos lineares (FASE 2 - V12)
            'Logistic Regression': {
                'C': Real(1e-4, 1e4, prior='log-uniform'),
                'penalty': Categorical(['l2', 'none']),
                'solver': Categorical(['lbfgs', 'saga'])
            },
            'LDA': {
                'solver': Categorical(['svd', 'lsqr', 'eigen']),
                'shrinkage': Categorical([None, 'auto', 0.1, 0.5, 0.9])
            }
            # Nota: QDA e Naive Bayes têm poucos hiperparâmetros,
            # então não incluímos espaços de busca para eles
        }
        self.trained_models = {}; self.scaler = StandardScaler()

    def get_available_models(self) -> list:
        return list(self.models.keys())

    def train_model(self, model_name: str, data: Dict[str, Any], data_summary: Dict[str, Any], 
                    selected_sensors: List[str], selected_classes: List[str], 
                    hyperparams: Dict[str, Any], use_optimization: bool = False) -> Dict[str, Any]:
        
        if model_name not in self.models: raise ValueError(f"Modelo {model_name} não disponível.")

        X_train_full, y_train_full = data['X_train'], data['y_train']; X_val_full, y_val_full = data['X_val'], data['y_val']; X_test_full, y_test_full = data['X_test'], data['y_test']
        train_mask = np.isin(y_train_full, selected_classes); X_train, y_train = X_train_full[train_mask], y_train_full[train_mask]
        val_mask = np.isin(y_val_full, selected_classes); X_val, y_val = X_val_full[val_mask], y_val_full[val_mask]
        test_mask = np.isin(y_test_full, selected_classes); X_test, y_test = X_test_full[test_mask], y_test_full[test_mask]
        all_sensors = data_summary['statistics']['Sensores Utilizados']
        points_per_sensor = data_summary['statistics']['Tamanho Final da Assinatura'] // len(all_sensors)
        X_train = _filter_data_by_sensors(X_train, all_sensors, selected_sensors, points_per_sensor)
        X_val = _filter_data_by_sensors(X_val, all_sensors, selected_sensors, points_per_sensor)
        X_test = _filter_data_by_sensors(X_test, all_sensors, selected_sensors, points_per_sensor)

        # ✅ NORMALIZAÇÃO: Modelos que precisam de escalonamento de features
        models_requiring_scaling = ['SVM', 'k-NN', 'MLP', 'Logistic Regression', 'Naive Bayes']
        if model_name in models_requiring_scaling:
            X_train = self.scaler.fit_transform(X_train)
            if X_val.size > 0: X_val = self.scaler.transform(X_val)
            if X_test.size > 0: X_test = self.scaler.transform(X_test)

        model = self.models[model_name]
        best_params = hyperparams

        if use_optimization and SKOPT_AVAILABLE and model_name in self.search_spaces:
            print(f"\n--- INICIANDO OTIMIZAÇÃO BAYESIANA PARA {model_name} ---")
            
            search_spaces = self.search_spaces[model_name]

            # --- ALTERAÇÃO: Ajuste dinâmico para o k-NN ---
            if model_name == 'k-NN':
                n_samples_train = X_train.shape[0]
                # O número de vizinhos não pode ser maior que o número de amostras
                # Usamos n_samples_train - 1 como um limite seguro
                # With cv=3, each fold trains on 2/3 of samples
                upper_bound = max(1, int(n_samples_train * 2 / 3) - 1)
                
                # Clona o dicionário para não alterar o original
                search_spaces = search_spaces.copy()
                # Atualiza o espaço de busca apenas para esta execução
                search_spaces['n_neighbors'] = Integer(1, upper_bound)
                print(f"Espaço de busca para 'n_neighbors' ajustado para (1, {upper_bound}) com base no tamanho dos dados.")

            opt = BayesSearchCV(estimator=model, search_spaces=search_spaces, n_iter=32, cv=3, n_jobs=-1, random_state=42)
            # Cast to plain Python strings to avoid numpy 2.x np.str_ repr breaking BayesSearchCV
            y_train_fit = np.array([str(lbl) for lbl in y_train])
            opt.fit(X_train, y_train_fit)
            final_model = opt.best_estimator_
            best_params = opt.best_params_
            print(f"--- MELHORES PARÂMETROS ENCONTRADOS: {best_params} ---")
        else:
            if use_optimization:
                print(f"AVISO: Otimização para {model_name} não executada. Verifique se 'scikit-optimize' está instalado e se o modelo é suportado.")
            final_model = model; final_model.set_params(**hyperparams); final_model.fit(X_train, y_train)
        
        # ✅ V12.1: Recebe também y_test e y_pred
        metrics, y_test_stored, y_pred_stored = self._evaluate(final_model, X_train, y_train, X_val, y_val, X_test, y_test)
        result = {
            'model': final_model,
            'model_name': model_name,
            'metrics': metrics,
            'best_params': best_params,
            'used_optimization': use_optimization,
            'selected_sensors': selected_sensors,
            'selected_classes': selected_classes,
            # ✅ Adiciona dados para visualizações
            'y_test': y_test_stored,
            'y_pred': y_pred_stored
        }
        self.trained_models[model_name] = result
        return result

    def _evaluate(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        metrics = {}
        # ✅ V12.1: Armazena y_test e y_pred para visualizações
        y_test_stored = None
        y_pred_stored = None

        if X_train.size > 0: metrics['train_accuracy'] = accuracy_score(y_train, model.predict(X_train))
        if X_val.size > 0: metrics['val_accuracy'] = accuracy_score(y_val, model.predict(X_val))
        if X_test.size > 0:
            y_test_pred = model.predict(X_test)

            # ✅ Armazena para visualizações posteriores
            y_test_stored = y_test.copy()
            y_pred_stored = y_test_pred.copy()

            metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
            metrics['precision'] = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_test_pred)
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

        # ✅ Retorna também os dados de teste e predições
        return metrics, y_test_stored, y_pred_stored