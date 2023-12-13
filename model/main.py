import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
import os

from process import Processing


class MainPoI:
    def __init__(self):
        pass

    def start(self):
        base_dir = "gowalla/"
        dataset_name = "gowalla"

        adjacency_matrix_filename = (
            "gowalla/adjacency_matrix_not_directed_48_7_categories_US.csv"
        )
        adjacency_matrix_week_filename = (
            "gowalla/adjacency_matrix_weekday_not_directed_48_7_categories_US.csv"
        )
        adjacency_matrix_weekend_filename = (
            "gowalla/adjacency_matrix_weekend_not_directed_48_7_categories_US.csv"
        )
        temporal_matrix_filename = (
            "gowalla/features_matrix_not_directed_48_7_categories_US.csv"
        )
        temporal_matrix_week_filename = (
            "gowalla/features_matrix_weekday_not_directed_48_7_categories_US.csv"
        )
        temporal_matrix_weekend_filename = (
            "gowalla/features_matrix_weekend_not_directed_48_7_categories_US.csv"
        )
        distance_matrix_filename = (
            "gowalla/distance_matrix_not_directed_48_7_categories_US.csv"
        )
        duration_matrix_filename = (
            "gowalla/duration_matrix_not_directed_48_7_categories_US.csv"
        )
        location_location_filename = (
            "gowalla/location_location_pmi_matrix_7_categories_US.npz"
        )
        location_time_filename = "gowalla/location_time_pmi_matrix_7_categories_US.csv"
        int_to_locationid_filename = "gowalla/int_to_locationid_7_categories_US.csv"

        country = "US"
        version = "normal"
        print("\nDataset: ", dataset_name)

        max_size_matrices = 3
        max_size_paths = 15
        n_splits = 5
        n_replications = 1
        epochs = 12

        self.GOWALLA_7_CATEGORIES = {
            "Shopping": 0,
            "Community": 1,
            "Food": 2,
            "Entertainment": 3,
            "Travel": 4,
            "Outdoors": 5,
            "Nightlife": 6,
        }
        int_to_category = {
            str(i): list(self.GOWALLA_7_CATEGORIES.keys())[i]
            for i in range(len(list(self.GOWALLA_7_CATEGORIES)))
        }


        output_dir = "./output/" + dataset_name + "/"

        if os.path.exists(output_dir):
            os.path.mkdir(output_dir)

        base_report = (
            "report_7_int_categories",
            {
                "0": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "1": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "2": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "3": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "4": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "5": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "6": {"precision": [], "recall": [], "f1-score": [], "support": []},
                "accuracy": [],
                "macro avg": {
                    "precision": [],
                    "recall": [],
                    "f1-score": [],
                    "support": [],
                },
                "weighted avg": {
                    "precision": [],
                    "recall": [],
                    "f1-score": [],
                    "support": [],
                },
            },
            "report",
        )

        adjacency_df = pd.read_csv(adjacency_matrix_filename).drop_duplicates(subset=['user_id'])
        temporal_df = pd.read_csv(temporal_matrix_filename).drop_duplicates(subset=['user_id'])
        distance_df = pd.read_csv(distance_matrix_filename).drop_duplicates(subset=['user_id'])
        duration_df = pd.read_csv(duration_matrix_filename).drop_duplicates(subset=['user_id'])

        adjacency_week_df = pd.read_csv(adjacency_matrix_week_filename).drop_duplicates(subset=['user_id'])
        temporal_week_df = pd.read_csv(temporal_matrix_week_filename).drop_duplicates(subset=['user_id'])

        adjacency_weekend_df = pd.read_csv(adjacency_matrix_weekend_filename).drop_duplicates(subset=['user_id'])
        temporal_weekend_df = pd.read_csv(temporal_matrix_weekend_filename).drop_duplicates(subset=['user_id'])

        print("\nVerificação de matrizes\n")
        self.matrices_verification(
            [
                adjacency_df,
                temporal_df,
                adjacency_week_df,
                temporal_week_df,
                adjacency_weekend_df,
                temporal_weekend_df,
                distance_df,
                duration_df,
            ]
        )

        location_location = sparse.load_npz(location_location_filename)
        location_time = pd.read_csv(location_time_filename)
        int_to_locationid = pd.read_csv(int_to_locationid_filename)


        inputs = {
            "all_week": {
                "adjacency": adjacency_df,
                "temporal": temporal_df,
                "distance": distance_df,
                "duration": duration_df,
                "location_location": location_location,
                "location_time": location_time,
                "int_to_locationid": int_to_locationid,
            },
            "week": {"adjacency": adjacency_week_df, "temporal": temporal_week_df},
            "weekend": {
                "adjacency": adjacency_weekend_df,
                "temporal": temporal_weekend_df,
            },
        }

        print("\nPreprocessing\n")
        (
            users_categories,
            adjacency_df,
            temporal_df,
            distance_df,
            duration_df,
            adjacency_week_df,
            temporal_week_df,
            adjacency_weekend_df,
            temporal_weekend_df,
            location_time_df,
            location_location_df,
            selected_users,
            df_selected_users_visited_locations,
        ) = Processing.poi_gnn_adjacency_preprocessing(
            inputs, max_size_matrices, True, True, 7, dataset_name
        )

        selected_users = pd.DataFrame({"selected_users": selected_users})

        inputs = {
            "all_week": {
                "adjacency": adjacency_df,
                "temporal": temporal_df,
                "location_time": location_time_df,
                "location_location": location_location_df,
                "categories": users_categories,
                "distance": distance_df,
                "duration": duration_df,
            },
            "week": {
                "adjacency": adjacency_week_df,
                "temporal": temporal_week_df,
                "categories": users_categories,
            },
            "weekend": {
                "adjacency": adjacency_weekend_df,
                "temporal": temporal_weekend_df,
                "categories": users_categories,
            },
        }

        usuarios = len(adjacency_df)

        folds, class_weight = Processing.k_fold_split_train_test(
            max_size_matrices, inputs, n_splits, "all_week"
        )

        (
            folds_week,
            class_weight_week,
        ) = Processing.k_fold_split_train_test(
            max_size_matrices, inputs, n_splits, "week"
        )

        (
            folds_weekend,
            class_weight_weekend,
        ) = Processing.k_fold_split_train_test(
            max_size_matrices, inputs, n_splits, "weekend"
        )

        print("\nclass weight: ", class_weight)
        inputs_folds = {
            "all_week": {"folds": folds, "class_weight": class_weight},
            "week": {"folds": folds_week, "class_weight": class_weight_week},
            "weekend": {"folds": folds_weekend, "class_weight": class_weight_weekend},
        }

        print("\nTreino\n")
        (
            folds_histories,
            base_report,
            model,
        ) = Processing.k_fold_with_replication_train_and_evaluate_model(
            inputs_folds,
            n_replications,
            max_size_matrices,
            max_size_paths,
            base_report,
            epochs,
            class_weight,
            country,
            version,
            output_dir,
        )

        selected_users.to_csv(output_dir + "selected_users.csv", index=False)
        print("\nbase: ", base_dir)
        base_report = Processing.preprocess_report(
            base_report, int_to_category
        )
        # self.poi_categorization_loader.plot_history_metrics(folds_histories, base_report, output_dir)
        self._save_report_to_csv(
            output_dir, base_report, n_splits, n_replications, usuarios
        )
        # self.poi_categorization_loader.save_model_and_weights(model, output_dir, n_splits, n_replications)
        print("\nUsuarios processados: ", usuarios)
    
    def _save_report_to_csv(self, output_dir, report, n_folds, n_replications, usuarios):

        precision_dict = {}
        recall_dict = {}
        fscore_dict = {}
        column_size = n_folds*n_replications
        for key in report:
            if key == 'accuracy':
                column = 'accuracy'
                fscore_dict[column] = report[key]
                continue
            elif key == 'recall' or key == 'f1-score' \
                    or key == 'support':
                continue
            if key == 'macro avg' or key == 'weighted avg':
                column = key
                fscore_dict[column] = report[key]['f1-score']
                continue
            fscore_column = key
            fscore_column_data = report[key]['f1-score']
            if len(fscore_column_data) < column_size:
                while len(fscore_column_data) < column_size:
                    fscore_column_data.append(np.nan)
            fscore_dict[fscore_column] = fscore_column_data

            precision_column = key
            precision_column_data = report[key]['precision']
            if len(precision_column_data) < column_size:
                while len(precision_column_data) < column_size:
                    precision_column_data.append(np.nan)
            precision_dict[precision_column] = precision_column_data

            recall_column = key
            recall_column_data = report[key]['recall']
            if len(recall_column_data) < column_size:
                while len(recall_column_data) < column_size:
                    recall_column_data.append(np.nan)
            recall_dict[recall_column] = recall_column_data

        precision = pd.DataFrame(precision_dict)
        print("Métricas precision: \n", precision)
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        precision.to_csv(output_dir + "precision.csv", index_label=False, index=False)

        recall = pd.DataFrame(recall_dict)
        print("Métricas recall: \n", recall)
        recall.to_csv(output_dir + "recall.csv", index_label=False, index=False)

        fscore = pd.DataFrame(fscore_dict)
        print("Métricas fscore: \n", fscore)
        fscore.to_csv(output_dir + "fscore.csv", index_label=False, index=False)

