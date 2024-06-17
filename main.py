import os
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from job.poi_categorization_job import PoiCategorizationJob
from configuration.poi_categorization_configuration import PoICategorizationConfiguration
from job.matrix_generation_for_poi_categorization_job import MatrixGenerationForPoiCategorizationJob
import mlflow 
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job',
        type=str,
        required=True,
        choices=['preprocess', 'categorize'],
        help="The job to run"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=False,
        help="The model to use"
    )

    parser.add_argument(
        '--base_line',
        type=str,
        required=False,
        help="baseline result path to compare"
    )

    parser.add_argument(
        '--base_line_general',
        type=str,
        required=False,
        help="baseline geral result path to compare"
    )

    parser.add_argument(
        '--state',
        type=str,
        required=True,
        help="data state"
    )
    
    args = parser.parse_args()
    
    

    if args.job == 'preprocess':
        job = MatrixGenerationForPoiCategorizationJob()
        job.start()
    
    else:
        if args.job == 'PoiCategorizationJob' and args.model is None:
            parser.error("--model is required for PoiCategorizationJob")

        if args.model not in ['pgc', 'havana', 'havana_arma', 'havana_gat', 'havana_no_agg']:
            parser.error("--model must be one of 'pgc', 'havana', 'havana_arma', 'havana_gat', 'havana_no_agg'")

        PoICategorizationConfiguration.MODEL = args.model
        PoICategorizationConfiguration.STATE = args.state
        PoICategorizationConfiguration.BASE_LINE = args.base_line
        PoICategorizationConfiguration.BASE_LINE_GENERAL = args.base_line_general

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("PoiCategorization")

        job = PoiCategorizationJob()
        job.start()
