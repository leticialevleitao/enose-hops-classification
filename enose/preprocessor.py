# src/preprocessor.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import f_oneway

class ENosePreprocessor:
    """
    Realiza o pré-processamento nos dados, com lógica de divisão adaptativa.
    """
    def _calculate_discriminatory_power(self, X_normalized, y, sensors_used, points_per_sensor, flatten_method):
        """
        Calcula o poder discriminatório (R²) de cada sensor.

        Args:
            X_normalized: Dados normalizados (num_amostras, features)
            y: Labels das classes
            sensors_used: Lista de nomes dos sensores
            points_per_sensor: Número de pontos temporais por sensor
            flatten_method: 'curve' ou 'interleave'

        Returns:
            Dicionário {sensor_name: r2_score} ordenado por R²
        """
        num_samples = X_normalized.shape[0]
        num_sensors = len(sensors_used)

        # Reconstrói a estrutura 3D baseado no método usado
        if flatten_method == 'curve':
            # CONCATENAÇÃO: [s1_completo, s2_completo, ...]
            # Reshape direto funciona
            deconstructed_data = X_normalized.reshape(num_samples, num_sensors, points_per_sensor)

        elif flatten_method == 'interleave':
            # ENTRELAÇAMENTO: [t0_todos, t1_todos, ...]
            # Precisa reorganizar primeiro
            total_points = num_sensors * points_per_sensor
            reshaped = X_normalized.reshape(num_samples, points_per_sensor, num_sensors)
            # Transpõe para (num_samples, num_sensors, points_per_sensor)
            deconstructed_data = reshaped.transpose(0, 2, 1)

        else:
            raise ValueError(f"Método desconhecido: {flatten_method}")

        r2_scores = {}
        for i, sensor_name in enumerate(sensors_used):
            # Extrai dados do sensor i
            sensor_data = deconstructed_data[:, i, :]  # (num_samples, points_per_sensor)

            # Calcula média temporal de cada amostra
            sensor_mean_responses = sensor_data.mean(axis=1)

            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                r2_scores[sensor_name] = 0.0
                continue

            # Agrupa por classe
            groups = [sensor_mean_responses[y == class_label] for class_label in unique_classes]
            groups = [g for g in groups if len(g) > 0]

            if len(groups) < 2:
                r2_scores[sensor_name] = 0.0
                continue

            # Calcula R² (variância entre classes / variância total)
            ss_total = np.sum((sensor_mean_responses - np.mean(sensor_mean_responses))**2)
            ss_within = sum(np.sum((group - np.mean(group))**2) for group in groups)

            if ss_total == 0:
                r2_scores[sensor_name] = 0.0
                continue

            r2 = (ss_total - ss_within) / ss_total
            r2_scores[sensor_name] = r2

        return dict(sorted(r2_scores.items(), key=lambda item: item[1], reverse=True))

    def process(self, data_summary: dict, normalization_method: str, split_ratios: tuple, flatten_method: str):
        X_raw = data_summary['all_data']
        y = data_summary['all_classes']
        stats = data_summary['statistics']
        sensors_used = stats['Sensores Utilizados']
        points_per_sensor = stats['Tamanho Final da Assinatura'] // len(sensors_used)
        
        deconstructed_cycles = X_raw.reshape(-1, len(sensors_used), points_per_sensor)

        # --- INÍCIO DA CORREÇÃO: LÓGICA DE DIVISÃO ADAPTATIVA ---
        
        train_ratio, val_ratio, test_ratio = split_ratios

        # 1. Primeira divisão (Treino vs. Temporário) - sempre estratificada
        cycles_train, cycles_temp, y_train, y_temp = train_test_split(
            deconstructed_cycles, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=42
        )
        
        # 2. Segunda divisão (Validação vs. Teste) - com verificação
        if val_ratio > 0 and len(y_temp) > 1:
            relative_test_size = test_ratio / (val_ratio + test_ratio)
            
            # Verifica se a estratificação é possível no conjunto temporário
            class_counts_temp = pd.Series(y_temp).value_counts()
            if class_counts_temp.min() < 2:
                print("AVISO: Não foi possível usar estratificação na divisão Val/Teste. Usando divisão normal.")
                # Fallback para divisão não estratificada se alguma classe tiver apenas 1 amostra
                cycles_val, cycles_test, y_val, y_test = train_test_split(
                    cycles_temp, y_temp, test_size=relative_test_size, random_state=42
                )
            else:
                # Divisão estratificada normal
                cycles_val, cycles_test, y_val, y_test = train_test_split(
                    cycles_temp, y_temp, test_size=relative_test_size, stratify=y_temp, random_state=42
                )
        else:
            cycles_val, y_val = np.array([]), np.array([])
            cycles_test, y_test = cycles_temp, y_temp

        # --- FIM DA CORREÇÃO ---

        # ✅ VALIDAÇÃO: Garantir mínimo de amostras por classe em cada conjunto
        min_samples_per_class = 2
        for dataset_name, y_data in [('Treino', y_train), ('Validação', y_val), ('Teste', y_test)]:
            if len(y_data) == 0:  # Ignora conjuntos vazios (ex: validação pode ser vazia)
                continue

            unique_classes, class_counts = np.unique(y_data, return_counts=True)
            problematic_classes = unique_classes[class_counts < min_samples_per_class]

            if len(problematic_classes) > 0:
                warning_msg = (
                    f"AVISO: Conjunto de {dataset_name} tem {len(problematic_classes)} classe(s) "
                    f"com menos de {min_samples_per_class} amostras: {problematic_classes.tolist()}\n"
                    f"Distribuição: {dict(zip(unique_classes, class_counts))}\n"
                    f"Considere: (1) Adicionar mais dados, (2) Remover classes pequenas, ou (3) Ajustar proporções treino/val/teste."
                )
                print(warning_msg)

        if normalization_method == 'minmax': scaler = MinMaxScaler()
        elif normalization_method == 'zscore': scaler = StandardScaler()
        else: raise ValueError("Método de normalização desconhecido.")
        
        if len(cycles_train) > 0:
            train_stack_for_fit = np.vstack([cycle.T for cycle in cycles_train])
            scaler.fit(train_stack_for_fit)
        
        def _transform_and_flatten(cycle_list, fitted_scaler, method):
            """
            Normaliza e achata os ciclos de acordo com o método escolhido.

            Args:
                cycle_list: Lista de ciclos com forma (num_sensores, pontos_por_sensor)
                fitted_scaler: Scaler já treinado
                method: 'curve' (concatenação) ou 'interleave' (entrelaçamento)

            Returns:
                Array 2D com assinaturas achatadas
            """
            if not isinstance(cycle_list, (list, np.ndarray)) or len(cycle_list) == 0:
                return np.array([])

            flattened_list = []
            for cycle in cycle_list:
                # cycle.shape = (num_sensores, pontos_por_sensor)

                # Normalização: transforma cada série temporal do sensor independentemente
                normalized_cycle_T = fitted_scaler.transform(cycle.T)  # (pontos, sensores)
                normalized_cycle = normalized_cycle_T.T  # Volta para (sensores, pontos)

                if method == 'curve':
                    # CONCATENAÇÃO DE CURVAS: Sensor por sensor
                    # normalized_cycle shape: (num_sensores, pontos_por_sensor)
                    # [s1_t0, s1_t1, ..., s1_tN, s2_t0, s2_t1, ..., s2_tN, ...]
                    signature = normalized_cycle.flatten(order='C')  # Row-major = linha por linha

                elif method == 'interleave':
                    # ENTRELAÇAMENTO POR INSTANTE: Todos os sensores em cada tempo
                    # [t0_s1, t0_s2, ..., t0_sN, t1_s1, t1_s2, ..., t1_sN, ...]
                    signature = normalized_cycle.flatten(order='F')  # Column-major = coluna por coluna

                else:
                    raise ValueError(f"Método de achatamento desconhecido: {method}")

                flattened_list.append(signature)

            return np.vstack(flattened_list)

        X_train = _transform_and_flatten(cycles_train, scaler, flatten_method)
        X_val = _transform_and_flatten(cycles_val, scaler, flatten_method)
        X_test = _transform_and_flatten(cycles_test, scaler, flatten_method)

        X_normalized = np.vstack([X_train, X_val, X_test] if X_val.size > 0 else [X_train, X_test])
        y_full = np.concatenate([y_train, y_val, y_test] if y_val.size > 0 else [y_train, y_test])

        r2_results = self._calculate_discriminatory_power(X_train, y_train, sensors_used, points_per_sensor, flatten_method)

        results = {
            'X_full_norm': X_normalized, 'y_full': y_full,
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'r2_scores': r2_results,
            'flatten_method': flatten_method,
            'points_per_sensor': points_per_sensor,
            'sensors_used': sensors_used,
            'normalization': normalization_method,  # ✅ Adiciona para inferência
            'stats': {
                'Total de Amostras': len(y), 'Amostras de Treino': len(y_train),
                'Amostras de Validação': len(y_val), 'Amostras de Teste': len(y_test)
            }
        }
        return results

    def process_single_signature(self, raw_data, normalization='minmax', flatten_method='interleave', fitted_scaler=None):
        """
        Processa uma única leitura (1 ciclo) do hardware para inferência.

        Args:
            raw_data: Dicionário {sensor_name: array_de_leituras}
            normalization: 'minmax' ou 'zscore' (usado apenas se fitted_scaler=None)
            flatten_method: 'curve' (concatenação) ou 'interleave' (entrelaçamento)
            fitted_scaler: Scaler já ajustado no conjunto de treino (recomendado).
                           Se None, um novo scaler é ajustado nesta amostra — isso
                           é cientificamente incorreto para inferência real e deve ser
                           evitado em produção.

        Returns:
            Array 1D com a assinatura normalizada e achatada
        """
        import warnings

        # Organiza dados por sensor
        sensors_ordered = sorted(raw_data.keys())
        sensor_arrays = [np.array(raw_data[sensor]) for sensor in sensors_ordered]

        # Empilha em formato (num_sensores, pontos_por_sensor)
        cycle = np.vstack(sensor_arrays)

        # Normaliza usando o scaler fornecido (correto para inferência)
        # ou um novo scaler por amostra (apenas para prototipagem rápida)
        if fitted_scaler is not None:
            normalized_cycle_T = fitted_scaler.transform(cycle.T)  # (pontos, sensores)
        else:
            warnings.warn(
                "process_single_signature: fitted_scaler não fornecido. "
                "Um scaler novo está sendo ajustado apenas nesta amostra. "
                "Isso não reproduz a distribuição do treino e é incorreto para inferência real.",
                UserWarning,
                stacklevel=2,
            )
            if normalization == 'minmax':
                scaler = MinMaxScaler()
            elif normalization == 'zscore':
                scaler = StandardScaler()
            else:
                raise ValueError(f"Método de normalização desconhecido: {normalization}")
            normalized_cycle_T = scaler.fit_transform(cycle.T)  # (pontos, sensores)

        normalized_cycle = normalized_cycle_T.T  # Volta para (sensores, pontos)

        # Achata de acordo com o método
        if flatten_method == 'curve':
            # CONCATENAÇÃO: [s1_completo, s2_completo, ...]
            signature = normalized_cycle.flatten(order='C')
        elif flatten_method == 'interleave':
            # ENTRELAÇAMENTO: [t0_todos, t1_todos, ...]
            signature = normalized_cycle.flatten(order='F')
        else:
            raise ValueError(f"Método de achatamento desconhecido: {flatten_method}")

        return signature