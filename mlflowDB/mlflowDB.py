import pandas as pd
from sklearn.metrics import classification_report
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import io
import tensorflow as tf
import numpy as np
from configuration.poi_categorization_configuration import PoICategorizationConfiguration
import os


state, run_name, base_line, base_line_general = None, None, None, None
def save(model, historys, data, accuracies):
    global run_name, state, base_line, base_line_general
    state, run_name, base_line, base_line_general = PoICategorizationConfiguration.STATE, PoICategorizationConfiguration.MODEL, PoICategorizationConfiguration.BASE_LINE, PoICategorizationConfiguration.BASE_LINE_GENERAL 
    os.makedirs('output/images', exist_ok=True)
    directory = f'output/{state}/general'
    os.makedirs(directory, exist_ok=True)

    save_history(historys)
    mlflow_metrics(data, accuracies)
    image_baseline_x_model_metrics(data, 'output/images/baseline_x_model_metrics.png')
    model_x_baseline_categories(
        data, accuracies, 'output/images/model_x_baseline_categories.png')
    mlflow.log_text(get_model_summary(model), "model_summary.txt")
    tf.keras.utils.plot_model(
        model, to_file="output/images/model.png", show_shapes=True)
    mlflow.log_artifact("output/images/model.png")
    mlflow.sklearn.log_model(model, "GNN")


def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def report_to_df_classes(data, accuracies):
    categories = ['Shopping', 'Community', 'Food',
                  'Entertainment', 'Travel', 'Outdoors', 'Nightlife']
    metrics = ['precision', 'recall', 'f1-score', 'support']

    df = pd.DataFrame(columns=metrics + ['category'])

    df['category'] = df['category'].astype('int32')
    df['support'] = df['support'].astype('int32')

    for class_label in ['0', '1', '2', '3', '4', '5', '6']:
        for i, linha in enumerate(zip(data[class_label]['precision'],
                                      data[class_label]['recall'],
                                      data[class_label]['f1-score'],
                                      data[class_label]['support'])):
            new_row = pd.DataFrame([{
                'precision': linha[0],
                'recall': linha[1],
                'f1-score': linha[2],
                'support': linha[3],
                'category': class_label,
                'accuracy': accuracies[i],
            }])

            df = pd.concat([df, new_row], ignore_index=True)

    df['category'] = df.apply(
        lambda row: categories[int(row['category'])], axis=1)

    return df


def mlflow_metrics(data, accuracies):
    df = report_to_df_classes(data, accuracies)

    metrics_dict = df.describe()

    for idx, metric_values in metrics_dict.items():
        for metric, value in metric_values.items():
            if metric != 'count':
                mlflow.log_metric(idx+'_'+metric.replace('%', '_pct'), value)


def save_history(historys):

    history = get_median_history(historys)

    for metric_name, h in history.items():
        for epoch, value in enumerate(h):
            mlflow.log_metric(metric_name, value, step=epoch)


def get_history_dict(history):
    hist = {}
    for metrics in history:
        for metric_name, value in metrics.items():
            metric_name = metric_name.split('_')
            if metric_name[0] == 'val':
                metric_name = '_'.join(metric_name[1:]) + '_val'
            else:
                metric_name = '_'.join(metric_name) + '_train'

            hist[metric_name] = np.array(value)

    history = hist

    return history


def get_median_history(historys):

    historys = [get_history_dict(history) for history in historys]
    median_history = {}
    div = {}
    for metric in historys[0].keys():
        shp = (max(len(fold[metric]) for fold in historys))
        median_history[metric] = np.zeros(shp)
        div[metric] = np.zeros(shp)
    for history in historys:
        for metric_name, value in history.items():
            div[metric_name][:len(value)] += 1
            median_history[metric_name][:len(value)] += value

    median_history['f1-score_val'] = (2 * median_history['precision_val'] * median_history['recall_val']) / (
        median_history['precision_val'] + median_history['recall_val'])
    median_history['f1-score_train'] = (2 * median_history['precision_train'] * median_history['recall_train']) / (
        median_history['precision_train'] + median_history['recall_train'])
    div['f1-score_val'] = 5
    div['f1-score_train'] = 5
    for metric_name, value in median_history.items():
        median_history[metric_name] /= div[metric_name]

    return median_history


def image_baseline_x_model_metrics(data, out):
    df = pd.DataFrame()
    global run_name, state, base_line_general
    for fold, (accuracy, macro, weighted) in enumerate(zip(data['accuracy'], data['macro avg']['f1-score'], data['weighted avg']['f1-score'])):
        new_line =  pd.DataFrame([{
            "fold": fold + 1,
            "Model": run_name,
            "accuracy": accuracy,
            "macro avg": macro,
            "weighted avg": weighted
        }])

        df = pd.concat([df, new_line], ignore_index=True)

    directory = f'output/{state}/general'
    file_path = f'{directory}/{run_name}.csv'
    df.to_csv(file_path, index=False)

    if(base_line_general):
        df_baseline = pd.read_csv(base_line_general , index_col=False)
        df = pd.concat([df, df_baseline], ignore_index=True)

    sns.color_palette("tab10")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    sns.boxplot(x='Model', y='accuracy', data=df, hue='Model')
    plt.title('Accuracy')
    plt.subplot(1, 3, 2)
    sns.boxplot(x='Model', y='macro avg', data=df, hue='Model')
    plt.title('Macro Avg')
    plt.subplot(1, 3, 3)
    sns.boxplot(x='Model', y='weighted avg', data=df, hue='Model')
    plt.title('Weighted Avg')
    plt.tight_layout()

    plt.savefig(out)
    mlflow.log_artifact(out)


def model_x_baseline_categories(data, accuracies, out):
    global run_name, state, base_line

    df = report_to_df_classes(data, accuracies)
    df["Model"] = run_name
    directory = f'output/{state}'
    file_path = f'{directory}/{run_name}.csv'

    df.to_csv(file_path, index=False)

    if(base_line):
        df_baseline = pd.read_csv(base_line, index_col=False)
        df = pd.concat([df, df_baseline], ignore_index=True)

    plt.figure(figsize=(15, 5))

    sns.boxplot(x='category', y='f1-score', hue='Model', data=df)

    plt.title(f"{'f1-score'.capitalize()} por Categoria")
    plt.xlabel('Category')
    plt.ylabel('f1-score'.capitalize())
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(out)
    mlflow.log_artifact(out)

