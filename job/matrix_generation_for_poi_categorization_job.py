import numpy as np
import pandas as pd

from domain.matrix_generation_for_poi_categorization_domain import MatrixGenerationForPoiCategorizationDomain
from extractor.file_extractor import FileExtractor
from configuration.poi_categorization_configuration import PoICategorizationConfiguration
from configuration.matrix_generation_for_poi_categorization_configuration import MatrixGenerationForPoiCategorizationConfiguration
from loader.poi_categorization_loader import PoiCategorizationLoader
from pathlib import Path

class MatrixGenerationForPoiCategorizationJob():

    def __init__(self):
        self.file_extractor = FileExtractor()
        self.matrix_generation_for_poi_categorization_domain = MatrixGenerationForPoiCategorizationDomain("gowalla")
        self.poi_categorization_loader = PoiCategorizationLoader()
        self.poi_categorization_configuration = PoICategorizationConfiguration()

    def start(self):
        osm_category_column = None
        users_checkin_filename = "gowalla/checkins.csv"
        adjacency_matrix_base_filename = "adjacency_matrix"
        features_matrix_base_filename = "features_matrix"
        sequence_matrix_base_filename = "sequence_matrix"
        distance_matrix_base_filename = "distance_matrix"
        duration_matrix_base_filename = "duration_matrix"
        pattern_matrices = "yes"
        directed = "no"
        top_users = "40000"
        dataset_name = "gowalla"
        categories_type = "7_categories"
        personal_matrix = "no"
        hour48 = "yes"
        base = "base"
        country = "United States"
        state = ""
        max_time_between_records = ""
        differemt_venues = "yes"
        print("\nDataset: ", dataset_name)

        convert_country = {'Brazil': 'BR', 'BR': 'BR', 'United States': 'US'}

        if personal_matrix == "no":
            personal_matrix = False
        else:
            personal_matrix = True
        if hour48 == "no":
            hour48 = False
            hour_file = "24_"
        else:
            hour48 = True
            hour_file = "48_"

        country_folder = convert_country[country] + "/"
        if len(state) > 0:
            state_folder = state + "/"
        else:
            state_folder = ""

        different_venues_dir = ""
        if differemt_venues == "yes":
            different_venues_dir = "different_venues/"
            differemt_venues = True
        else:
            differemt_venues = False

        userid_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['userid_column']
        category_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['category_column']
        category_name_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['category_name_column']
        locationid_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['locationid_column']
        datetime_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['datetime_column']
        latitude_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['latitude_column']
        longitude_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['longitude_column']
        country_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['country_column']
        state_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['state_column']
        num_users = MatrixGenerationForPoiCategorizationConfiguration.NUM_USERS.get_value()[dataset_name]
        category_to_int = self.poi_categorization_configuration.CATEGORIES_TO_INT[dataset_name][categories_type]
        max_time_between_records_dir = self.poi_categorization_configuration.MAX_TIME_BETWEEN_RECORDS[1][max_time_between_records]
        max_size_matrices = self.poi_categorization_configuration.MAX_SIZE_MATRICES[1]
        n_splits = self.poi_categorization_configuration.N_SPLITS[1]
        n_replications = self.poi_categorization_configuration.N_REPLICATIONS[1]

        output_base_dir = self.poi_categorization_configuration.OUTPUT_DIR[1]
        dataset_type_dir = self.poi_categorization_configuration.DATASET_TYPE[1][dataset_name]
        category_type_dir = self.poi_categorization_configuration.CATEGORY_TYPE[1][categories_type]
        output_dir = output_base_dir + dataset_type_dir + category_type_dir

        dtypes_columns = {userid_column: int, category_column: 'Int16', category_name_column: 'category',
                          locationid_column: 'category', datetime_column: 'category', latitude_column: 'float64',
                          longitude_column: 'float64'}


        users_checkin = self.file_extractor.read_csv(users_checkin_filename, dtypes_columns).query(country_column + " == '"+country+"'")
        if category_column == category_name_column:
            categories = users_checkin[category_name_column].tolist()
            categories_int = []
            for i in range(len(categories)):
                if categories[i] == 'Other':
                    categories_int.append(-1)
                else:
                    categories_int.append(category_to_int[categories[i]])

            category_column = category_column + "_id"
            users_checkin[category_column] = np.array(categories_int)
            print("\n\n", users_checkin[category_column], "\n\n", users_checkin[category_column].unique() )

        if state != "":
            users_checkin = users_checkin.query(state_column + " == '" + state + "'")
        print("\n----- verificação -----")
        print("\nPais: ", users_checkin[country_column].unique().tolist())
        if len(state) > 0:
            print("\nEstado: ", users_checkin[state_column].unique().tolist())

        # data

        users_checkin[datetime_column] = pd.to_datetime(users_checkin[datetime_column], infer_datetime_format=True)
        users_checkin[category_column] = users_checkin[category_column].astype('int')
        if dataset_name == 'raw_gps':
            personal_category_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['personal_category_column']
            users_checkin = users_checkin.query("" + personal_category_column + " != 'home'")
            # coluna com as categorias em um determinado raio em metros
            osm_category_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['osm_category_column']

        print("\ncoluna osm", osm_category_column)

        """
        Generate matrixes for each user 
        """
        if personal_matrix:
            directed = False
            folder = 'gowalla/'
            adjacency_matrix_base_filename = folder + adjacency_matrix_base_filename + "not_directed_personal_" + hour_file + categories_type + ".csv"
            features_matrix_base_filename = folder + features_matrix_base_filename + "not_directed_personal_" + hour_file + categories_type + ".csv"
            sequence_matrix_base_filename = folder + sequence_matrix_base_filename + "not_directed_personal_" + hour_file + categories_type + ".csv"
        elif directed == "no":
            directed = False
            folder = 'gowalla/'
            self.folder_generation(folder)
            country = convert_country[country]
            adjacency_matrix_filename = folder + adjacency_matrix_base_filename + "_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            adjacency_weekday_matrix_filename = folder + adjacency_matrix_base_filename + "_weekday_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            adjacency_weekend_matrix_filename = folder+adjacency_matrix_base_filename + "_weekend_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            temporal_matrix_filename = folder+features_matrix_base_filename + "_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            temporal_weekday_matrix_filename = folder+features_matrix_base_filename + "_weekday_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            temporal_weekend_matrix_filename = folder+features_matrix_base_filename + "_weekend_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            path_matrix_filename = folder+sequence_matrix_base_filename + "_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            path_weekeday_matrix_filename = folder+sequence_matrix_base_filename + "_weekday_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            path_weekend_matrix_filename = folder+sequence_matrix_base_filename + "_weekend_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            distance_matrix_filename = folder + distance_matrix_base_filename + "_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            distance_weekday_matrix_filename = folder + distance_matrix_base_filename + "_weekday_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            distance_weekend_matrix_filename = folder + distance_matrix_base_filename + "_weekend_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            duration_matrix_filename = folder + duration_matrix_base_filename + "_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            duration_weekday_matrix_filename = folder + duration_matrix_base_filename + "_weekday_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            duration_weekend_matrix_filename = folder + duration_matrix_base_filename + "_weekend_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            location_locaion_pmi_matrix_filename = folder + "location_location_pmi_matrix_" + categories_type + "_" + country + ".npz"
            location_time_pmi_matrix_filename = folder + "location_time_pmi_matrix_" + categories_type + "_" + country + ".csv"
            int_to_locationid_filename = folder + "int_to_locationid_" + categories_type + "_" + country + ".csv"
        else:
            directed = True
            folder = 'gowalla/'
            adjacency_matrix_base_filename = folder+adjacency_matrix_base_filename + "directed_"+hour_file+categories_type+"_"+country+".csv"
            distance_matrix_base_filename = folder+distance_matrix_base_filename + "directed_"+hour_file+categories_type+"_"+country+".csv"
            user_poi_vector_base_filename = folder+"user_poi_vector_directed7_categories_United States.csv"
            sequence_matrix_base_filename = folder + sequence_matrix_base_filename + "directed_" + hour_file + categories_type +"_"+country+ ".csv"

        if pattern_matrices == "yes":
            self.matrix_generation_for_poi_categorization_domain\
                .generate_pattern_matrices(users_checkin,
                                           dataset_name,
                                           categories_type,
                                           adjacency_matrix_filename,
                                           adjacency_weekday_matrix_filename,
                                           adjacency_weekend_matrix_filename,
                                           temporal_matrix_filename,
                                           temporal_weekday_matrix_filename,
                                           temporal_weekend_matrix_filename,
                                           distance_matrix_filename,
                                           duration_matrix_filename,
                                           location_locaion_pmi_matrix_filename,
                                           location_time_pmi_matrix_filename,
                                           int_to_locationid_filename,
                                           userid_column,
                                           category_column,
                                           locationid_column,
                                           latitude_column,
                                           longitude_column,
                                           datetime_column,
                                           differemt_venues,
                                           directed,
                                           personal_matrix,
                                           top_users,
                                           max_time_between_records,
                                           num_users,
                                           base,
                                           hour48,
                                           osm_category_column)
        else:
            self.matrix_generation_for_poi_categorization_domain \
                .generate_gpr_matrices_v2(users_checkin,
                                          dataset_name,
                                           adjacency_matrix_base_filename,
                                           distance_matrix_base_filename,
                                          user_poi_vector_base_filename,
                                           userid_column,
                                          category_column,
                                          locationid_column,
                                           latitude_column,
                                           longitude_column,
                                           datetime_column)

    def folder_generation(self, folder):
        Path(folder).mkdir(parents=True, exist_ok=True)