# enose/data_loader.py

import os
import numpy as np
import pandas as pd

COLUMNS_NAME = ["0","1","timestamp","TGS826","TGS2611","TGS2603","TGS813","TGS822","TGS2602","TGS823","temperature","pressao","humidity"]
SENSOR_COLUMNS = ["TGS826","TGS2611","TGS2603","TGS813","TGS822","TGS2602","TGS823"]

class DataLoadingWorker:
    def __init__(self, file_paths: list, selected_sensors: list):
        self.file_paths = file_paths
        self.selected_sensors = selected_sensors
        self.is_running = True

    def run(self):
        all_cycles_raw_data = []
        all_classes = []
        file_info = []

        for i, file_path in enumerate(self.file_paths):
            if not self.is_running: break

            base_name = os.path.basename(file_path)
            class_name, _ = os.path.splitext(base_name)

            with open(file_path, 'r') as f:
                current_cycle_data = []
                is_in_cycle = False
                for line in f:
                    if "_CICLO_INICIADO_" in line:
                        is_in_cycle = True
                        current_cycle_data = []
                        continue
                    if "_CICLO_FINALIZADO_" in line:
                        is_in_cycle = False
                        if current_cycle_data:
                            all_cycles_raw_data.append(np.array(current_cycle_data))
                            all_classes.append(class_name)
                        continue

                    if is_in_cycle and len(line.strip()) > 1:
                        try:
                            values = [float(s) for s in line.split()]
                            all_sensor_values = values[3:3 + len(SENSOR_COLUMNS)]
                            current_cycle_data.append(all_sensor_values)
                        except (ValueError, IndexError):
                            continue

            file_info.append({"filename": base_name, "class_name": class_name})

        if not all_cycles_raw_data:
            raise ValueError("No valid cycles found in the selected files.")

        cycle_lengths = [len(cycle) for cycle in all_cycles_raw_data]
        min_len = min(cycle_lengths)
        max_len = max(cycle_lengths)
        mean_len = np.mean(cycle_lengths)
        TARGET_CYCLE_LENGTH = min_len

        standardized_cycles = []
        for cycle_data in all_cycles_raw_data:
            truncated_cycle = cycle_data[:TARGET_CYCLE_LENGTH, :]
            standardized_cycles.append(truncated_cycle)

        sensor_indices_to_keep = [i for i, name in enumerate(SENSOR_COLUMNS) if name in self.selected_sensors]

        all_signatures = []
        for std_cycle in standardized_cycles:
            filtered_cycle = std_cycle[:, sensor_indices_to_keep]
            signature = filtered_cycle.flatten(order='F')
            all_signatures.append(signature)

        all_signatures_np = np.vstack(all_signatures)
        all_classes_np = np.array(all_classes)

        points_per_sensor = TARGET_CYCLE_LENGTH
        preview_columns = []
        for sensor_name in self.selected_sensors:
            preview_columns.extend([f"{sensor_name}_{i+1}" for i in range(points_per_sensor)])

        preview_df = pd.DataFrame(all_signatures_np, columns=preview_columns)
        preview_df['Classe'] = all_classes_np

        amostras_por_classe_dict = pd.Series(all_classes_np).value_counts().to_dict()
        amostras_por_classe_limpo = {k: int(v) for k, v in amostras_por_classe_dict.items()}

        num_selected_sensors = len(self.selected_sensors)
        struct_string = f"{num_selected_sensors} sensores x {points_per_sensor} pontos/ciclo = {all_signatures_np.shape[1]} colunas"

        summary = {
            'all_data': all_signatures_np,
            'all_classes': all_classes_np,
            'preview_df': preview_df,
            'statistics': {
                'Total de Amostras (Ciclos)': len(all_signatures_np),
                'Sensores Utilizados': self.selected_sensors,
                'Comprimento Mínimo dos Ciclos': int(min_len),
                'Comprimento Máximo dos Ciclos': int(max_len),
                'Comprimento Médio dos Ciclos': f"{mean_len:.2f}",
                'Tamanho Final da Assinatura': all_signatures_np.shape[1],
                'Estrutura da Assinatura': struct_string,
                'Amostras por Classe': amostras_por_classe_limpo
            }
        }

        return summary

    def stop(self):
        self.is_running = False
