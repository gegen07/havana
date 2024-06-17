import math
import pandas as pd
import numpy as np
import time
import statistics as st
import os
from scipy.sparse import dok_matrix
from numpy.linalg import norm
from numpy.linalg import inv as inverse
import scipy.sparse as sparse
from sklearn.decomposition import NMF
from configuration.weekday  import Weekday

from foundation.util.geospatial_utils import points_distance
from loader.matrix_generation_for_poi_categorization_loarder import MatrixGenerationForPoiCategorizationLoader
from configuration.poi_categorization_configuration import PoICategorizationConfiguration
from extractor.file_extractor import FileExtractor


class MatrixGenerationForPoiCategorizationDomain:


    def __init__(self, dataset_name):
        self.matrix_generation_for_poi_categorization_loader = MatrixGenerationForPoiCategorizationLoader()
        self.file_extractor = FileExtractor()
        self.poi_categorization_configuration = PoICategorizationConfiguration()
        self.dataset_name = dataset_name
        self.distance_sigma = 10
        self.duration_sigma = 10
        self.max_events = 200
        self.count_usuarios = 0
        self.anterior = 0
        self.LL = np.array([])
        self.LT = np.array([])

    def filter_user(self, user_checkin,
                    dataset_name,
                    userid_column,
                   userid,
                   datetime_column,
                   category_column):

        user_checkin = user_checkin.sort_values(by=[datetime_column]).head(self.max_events)
        categories = user_checkin[category_column].tolist()
        if dataset_name == "gowalla":
            if len(user_checkin[category_column].unique().tolist()) < 7:
                return pd.DataFrame({'tipo': ['nan'], userid_column: [userid]})
        elif dataset_name == "user_tracking":
                return pd.DataFrame({'tipo': ['bom'], userid_column: [userid]})

    def generate_user_matrices(self, user_checkin,
                               userid,
                               datetime_column,
                               locationid_column,
                               category_column,
                               latitude_column,
                               longitude_column,
                               osm_category_column,
                               dataset_name,
                               categories_type,
                               base,
                               files_names,
                               differemt_venues, personal_features_matrix, hour48, directed, max_time_between_records):
        """
        :param user_checkin:
        :param datetime_column:
        :param locationid_column:
        :param category_column:
        :param osm_category_column:
        :param personal_features_matrix:
        :param hour48:
        :param directed:
        :return: adjacency, temporal, and path matrices
        """

        user_checkin = user_checkin.sort_values(by=[datetime_column])
        latitude_list = user_checkin[latitude_column].tolist()
        longitude_list = user_checkin[longitude_column].tolist()

        user_checkin[category_column] = np.array([self.poi_categorization_configuration.GOWALLA_7_CATEGORIES_TO_INT[i] for i in user_checkin['category'].tolist()])
                
        # matrices initialization
        visited_location_ids = user_checkin[locationid_column].tolist()
        visited_location_ids_real = []
        n_pois = len(user_checkin[locationid_column].unique().tolist())
        adjacency_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        adjacency_weekday_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        adjacency_weekend_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        temporal_weekday_matrix = [[0 for i in range(24)] for j in range(n_pois)]
        temporal_weekend_matrix = [[0 for i in range(24)] for j in range(n_pois)]
        distance_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        duration_matrix = [[[] for i in range(n_pois)] for j in range(n_pois)]
        if personal_features_matrix or not hour48:
            temporal_matrix = [[0 for i in range(24)] for j in range(n_pois)]
        else:
            temporal_matrix = [[0 for i in range(48)] for j in range(n_pois)]
        categories_list = [-1 for i in range(n_pois)]

        datetimes = user_checkin[datetime_column].tolist()
        placeids = user_checkin[locationid_column].tolist()
        placeids_unique = user_checkin[locationid_column].unique().tolist()
        placeids_unique_to_int = {placeids_unique[i]: i for i in range(len(placeids_unique))}
        # converter os ids dos locais para inteiro
        placeids_int = [placeids_unique_to_int[placeids[i]] for i in range(len(placeids))]
        categories = user_checkin[category_column].tolist()
        if base != "predict":
            min_categories_ = {'gowalla': 5, 'user_tracking': 1}
        else:
            min_categories_ = {'gowalla': 5, 'user_tracking': 2}
        min_categories = min_categories_[dataset_name]
        if not personal_features_matrix:
            if hour48:
                if datetimes[0].weekday() < 5:
                    hour = datetimes[0].hour
                    temporal_matrix[placeids_int[0]][hour] += 1
                    temporal_weekday_matrix[placeids_int[0]][hour] += 1
                else:
                    hour = datetimes[0].hour + 24
                    temporal_matrix[placeids_int[0]][hour] += 1
                    temporal_weekend_matrix[placeids_int[0]][hour - 24] += 1
            else:
                hour = datetimes[0].hour
                temporal_matrix[placeids_int[0]][hour] += 1
        else:
            if datetimes[0].weekday() < 5:
                temporal_matrix[placeids_int[0]][math.floor(datetimes[0].hour / 2)] += 1
            else:
                temporal_matrix[placeids_int[0]][math.floor(datetimes[0].hour / 2) + 12] += 1
        categories_list[0] = categories[0]

        count = 0
        max_timedelta = pd.Timedelta(days=2020)
        if len(max_time_between_records) > 0:
            max_timedelta = pd.Timedelta(days=int(max_time_between_records))
        for j in range(1, len(datetimes)):
            anterior = j - 1
            atual = j
            local_anterior = placeids_int[anterior]
            local_atual = placeids_int[atual]
            lat_before = latitude_list[anterior]
            lng_before = longitude_list[anterior]
            lat_current = latitude_list[atual]
            lng_current = longitude_list[atual]
            if distance_matrix[local_anterior][local_atual] == 0:
                distance = int(points_distance([lat_before, lng_before], [lat_current, lng_current]) / 1000)
                distance = self._distance_importance(distance)
            else:
                distance = distance_matrix[local_anterior][local_atual]

            datetime_before = datetimes[anterior]
            datetime_current = datetimes[atual]
            duration = int((datetime_current - datetime_before).total_seconds() / 3600)
            duration = self._duration_importance(duration)
            distance_matrix[local_anterior][local_atual] = distance
            distance_matrix[local_atual][local_anterior] = distance
            duration_matrix[local_anterior][local_atual].append(duration)

            if directed:
                adjacency_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                if datetimes[atual].weekday() < 5:
                    adjacency_weekday_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                else:
                    adjacency_weekend_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
            else:
                adjacency_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                adjacency_matrix[placeids_int[atual]][placeids_int[anterior]] += 1

                if datetimes[atual].weekday() < 5:
                    adjacency_weekday_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                    adjacency_weekday_matrix[placeids_int[atual]][placeids_int[anterior]] += 1
                else:
                    adjacency_weekend_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                    adjacency_weekend_matrix[placeids_int[atual]][placeids_int[anterior]] += 1

            visited_location_ids_real.append(visited_location_ids[atual])

            if not personal_features_matrix:
                if hour48:
                    if datetimes[atual].weekday() < 5:
                        hour = datetimes[atual].hour
                        temporal_matrix[placeids_int[atual]][hour] += 1
                        temporal_weekday_matrix[placeids_int[atual]][hour] += 1
                    else:
                        hour = datetimes[atual].hour + 24
                        temporal_matrix[placeids_int[atual]][hour] += 1
                        temporal_weekend_matrix[placeids_int[atual]][hour - 24] += 1
                else:
                    hour = datetimes[atual].hour
                    temporal_matrix[placeids_int[atual]][hour] += 1
            else:
                if datetimes[atual].weekday() < 5:
                    temporal_matrix[placeids_int[atual]][math.floor(datetimes[atual].hour / 2)] += 1
                else:
                    temporal_matrix[placeids_int[atual]][math.floor(datetimes[atual].hour / 2) + 12] += 1

            categories_list[placeids_int[atual]] = categories[atual]


        if osm_category_column is not None:
            # pre-processar raw gps
            adjacency_matrix, temporal_matrix, categories_list = self.remove_raw_gps_pois_that_dont_have_categories(
                categories_list, adjacency_matrix, temporal_matrix)
            if adjacency_matrix != []:
                count += 1

        else:
            adjacency_matrix, features_matrix, categories_list = self.remove_gps_pois_that_dont_have_categories(
                categories_list, adjacency_matrix, temporal_matrix)
            adjacency_weekday_matrix, features_weekday_matrix, categories_weekday_list = self.remove_gps_pois_that_dont_have_categories(
                categories_list, adjacency_weekday_matrix, temporal_weekday_matrix)
            adjacency_weekend_matrix, features_weekend_matrix, categories_list = self.remove_gps_pois_that_dont_have_categories(
                categories_list, adjacency_weekend_matrix, temporal_weekend_matrix)

            if len(adjacency_matrix) <= 2 or len(temporal_matrix) <= 2 or len(categories_list) <= 2:
                pass

        duration_matrix = self._summarize_categories_distance_matrix(duration_matrix)

        visited_location_ids_real = user_checkin[locationid_column].unique().tolist()

        if len(adjacency_matrix) < 2:
            print("\nUsuário com poucas categorias diferentes visitadas")
            return pd.DataFrame({'adjacency': ['vazio'], 'adjacency_weekday': ['vazio'],
                                 'adjacency_weekend': ['vazio'], 'temporal': ['vazio'],
                                 'distance': ['vazio'], 'duration': ['vazio'],
                                 'temporal_weekday': ['vazio'],
                                 'temporal_weekend': ['vazio'],
                                 'visited_location_ids': ['vazio'],
                                 'category': ['vazio']})

        columns = ["userid", "adjacency", "adjacency_weekday", "adjacency_weekend", "temporal", "distance", "duration",
                   "temporal_weekday", "temporal_weekend", "visited_location_ids", "category"]

        user_checkin = pd.DataFrame({'userid': [userid], 'adjacency': [adjacency_matrix], 'adjacency_weekday': [adjacency_weekday_matrix],
                             'adjacency_weekend': [adjacency_weekend_matrix], 'temporal': [temporal_matrix],
                             'distance': [distance_matrix], 'duration': [duration_matrix],
                             'temporal_weekday': [temporal_weekday_matrix], 'temporal_weekend': [temporal_weekend_matrix],
                             'visited_location_ids': [visited_location_ids_real],
                             'category': [categories_list]})

        user_checkin = user_checkin[columns]
        user_checkin.columns = np.array(
            ["userid", "adjacency", "adjacency_weekday", "adjacency_weekend", "temporal", "distance", "duration",
             "temporal_weekday", "temporal_weekend", "visited_location_ids", "category"])

        users_checkin = user_checkin[user_checkin['adjacency'] != 'vazio']
        adjacency_matrix_df = user_checkin[['userid', 'adjacency', 'category', 'visited_location_ids']]
        adjacency_weekday_matrix_df = user_checkin[['userid', 'adjacency_weekday', 'category']]
        adjacency_weekend_matrix_df = user_checkin[['userid', 'adjacency_weekend', 'category']]
        temporal_matrix_df = user_checkin[['userid', 'temporal', 'category']]
        temporal_weekday_matrix_df = user_checkin[['userid', 'temporal_weekday', 'category']]
        temporal_weekend_matrix_df = user_checkin[['userid', 'temporal_weekend', 'category']]
        distance_matrix_df = user_checkin[['userid', 'distance', 'category']]
        duration_matrix_df = user_checkin[['userid', 'duration', 'category']]

        adjacency_matrix_df.columns = ['user_id', 'matrices', 'category', 'visited_location_ids']
        adjacency_weekend_matrix_df.columns = ['user_id', 'matrices', 'category']
        adjacency_weekday_matrix_df.columns = ['user_id', 'matrices', 'category']
        temporal_matrix_df.columns = ['user_id', 'matrices', 'category']
        temporal_weekday_matrix_df.columns = ['user_id', 'matrices', 'category']
        temporal_weekend_matrix_df.columns = ['user_id', 'matrices', 'category']
        distance_matrix_df.columns = ['user_id', 'matrices', 'category']
        duration_matrix_df.columns = ['user_id', 'matrices', 'category']
        files = [adjacency_matrix_df,
                 adjacency_weekday_matrix_df,
                 adjacency_weekend_matrix_df,
                 temporal_matrix_df,
                 temporal_weekday_matrix_df,
                 temporal_weekend_matrix_df,
                 distance_matrix_df,
                 duration_matrix_df]

        self.matrix_generation_for_poi_categorization_loader. \
            adjacency_features_matrices_to_csv(files,
                                               files_names
                                               )

        self.count_usuarios += 1
        if self.count_usuarios > self.anterior + 100:
            self.anterior = self.count_usuarios
            print("\nNúmero de usuários: ", self.count_usuarios)
        return pd.DataFrame({'adjacency': ['vazio'], 'adjacency_weekday': ['vazio'],
                                 'adjacency_weekend': ['vazio'], 'temporal': ['vazio'],
                                 'distance': ['vazio'], 'duration': ['vazio'],
                                 'temporal_weekday': ['vazio'],
                                 'temporal_weekend': ['vazio'],
                                 'visited_location_ids': ['vazio'],
                                 'category': ['vazio']})

    def pmi(self, Dt):

        Dt = np.array(Dt)
        sum_of_dt = np.sum(Dt)
        l_occurrency = np.sum(Dt, axis=1)
        c_occurrency = np.sum(Dt, axis=0)

        size = len(Dt)
        sizes = len(Dt[0])
        for i in range(size):

            for j in range(sizes):

                p = (Dt[i, j] * len(Dt)) / (l_occurrency[i] * c_occurrency[j])
                if type(p) != float:
                    p = 1
                p = np.maximum(p, 1)

                Dt[i, j] = np.maximum(np.log2(p), 0)

        return Dt.tolist()

    def generate_gpr_user_matrices(self, user_checkin,
                               userid,
                                   datetime_column,
                                   locationid_column,
                                   category_column,
                                   latitude_column,
                                   longitude_column,
                                   dataset_name,
                                   files_names):
        """
        :param user_checkin:
        :param datetime_column:
        :param locationid_column:
        :param category_column:
        :param osm_category_column:
        :param personal_features_matrix:
        :param hour48:
        :param directed:
        :return: adjacency, temporal, and path matrices
        """

        user_checkin = user_checkin.sort_values(by=[datetime_column]).drop_duplicates()
        user_checkin = user_checkin.head(self.max_events)
        latitude_list = user_checkin[latitude_column].tolist()
        longitude_list = user_checkin[longitude_column].tolist()

        user_checkin[category_column] = np.array(
            [self.poi_categorization_configuration.CATEGORIES_TO_INT[dataset_name]['7_categories'][i] for i in user_checkin['category'].tolist()])

        # matrices initialization
        visited_location_ids = user_checkin[locationid_column].tolist()
        visited_location_ids_real = []
        n_pois = len(user_checkin[locationid_column].unique().tolist())
        adjacency_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        distance_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        categories_list = [-1 for i in range(n_pois)]
        user_poi_vector_column = []
        user_category_vector = [0] * 7

        datetimes = user_checkin[datetime_column].tolist()
        placeids = user_checkin[locationid_column].tolist()
        placeids_unique = user_checkin[locationid_column].unique().tolist()
        placeids_unique_to_int = {placeids_unique[i]: i for i in range(len(placeids_unique))}
        # converter os ids dos locais para inteiro
        placeids_int = [placeids_unique_to_int[placeids[i]] for i in range(len(placeids))]
        categories = user_checkin[category_column].tolist()

        categories_list[0] = categories[0]
        user_category_vector[categories[0]] = 1
        count = 0
        max_timedelta = pd.Timedelta(days=2020)
        for j in range(1, len(datetimes)):
            anterior = j - 1
            atual = j
            local_anterior = placeids_int[anterior]
            local_atual = placeids_int[atual]

            lat_before = latitude_list[anterior]
            lng_before = longitude_list[anterior]
            lat_current = latitude_list[atual]
            lng_current = longitude_list[atual]
            if distance_matrix[local_anterior][local_atual] == 0:
                distance = int(points_distance([lat_before, lng_before], [lat_current, lng_current]) / 1000)
                distance = self._distance_importance(distance)
            else:
                distance = distance_matrix[local_anterior][local_atual]

            datetime_before = datetimes[anterior]
            datetime_current = datetimes[atual]
            duration = int((datetime_current - datetime_before).total_seconds() / 3600)
            duration = self._duration_importance(duration)
            distance_matrix[local_anterior][local_atual] = distance
            distance_matrix[local_atual][local_anterior] = distance

            user_category_vector[categories[atual]] += 1

            adjacency_matrix[placeids_int[anterior]][placeids_int[atual]] += 1

            visited_location_ids_real.append(visited_location_ids[atual])

            categories_list[placeids_int[atual]] = categories[atual]

        visited_location_ids_real = user_checkin[locationid_column].unique().tolist()

        if len(adjacency_matrix) < 2:
            print("\nUsuário com poucas categorias diferentes visitadas")
            return pd.DataFrame({'adjacency': ['vazio'],
                                 'distance': ['vazio'],
                                 'user_poi': ['vazio'],
                                 'visited_location_ids': ['vazio'],
                                 'category': ['vazio']})

        columns = ["userid", "adjacency", "distance", "user_poi", "visited_location_ids", "category"]

        user_checkin = pd.DataFrame(
            {'userid': [userid], 'adjacency': [adjacency_matrix], 'distance': [distance_matrix], 'user_poi': [user_category_vector],
             'visited_location_ids': [visited_location_ids_real],
             'category': [categories_list]})

        user_checkin = user_checkin[columns]
        user_checkin.columns = np.array(
            ["userid", "adjacency", "distance",  "user_poi", "visited_location_ids", "category"])

        users_checkin = user_checkin[user_checkin['adjacency'] != 'vazio']
        adjacency_matrix_df = user_checkin[['userid', 'adjacency', 'category', 'visited_location_ids']]
        distance_matrix_df = user_checkin[['userid', 'distance', 'category']]
        user_poi_vector_df = user_checkin[['userid', 'user_poi', 'category']]

        adjacency_matrix_df.columns = ['user_id', 'matrices', 'category', 'visited_location_ids']
        distance_matrix_df.columns = ['user_id', 'matrices', 'category']
        user_poi_vector_df.columns = ['user_id', 'matrices', 'category']

        files = [adjacency_matrix_df,
                 distance_matrix_df,
                 user_poi_vector_df]

        self.matrix_generation_for_poi_categorization_loader. \
            adjacency_features_matrices_to_csv(files,
                                               files_names
                                               )

        self.count_usuarios += 1
        if self.count_usuarios > self.anterior + 100:
            self.anterior = self.count_usuarios
            print("\nUsuários: ", self.count_usuarios)

        return pd.DataFrame({'adjacency': ['vazio'],
                             'distance': ['vazio'],
                             'user_poi': ['vazio'],
                             'visited_location_ids': ['vazio'],
                             'category': ['vazio']})

    def reduce_user_data(self, user_checkin, datetime_column):

        user_checkin = user_checkin.sort_values(by=[datetime_column])
        user_checkin = user_checkin.head(self.max_events + 200)

        return user_checkin

    def _create_location_coocurrency_matrix(self, users_checkins, userid_column, datetime_column, locationid_column, locationid_to_int):
        try:

            users_checkins["time"] = [d.time() for d in users_checkins[datetime_column]]
            number_of_locations = len(users_checkins[locationid_column].unique())
            self.LL = sparse.lil_matrix(
                (number_of_locations, number_of_locations))  #location co occurency represents memory for save memory
            cont = 0
            init = time.time()
            for user_id in users_checkins[userid_column].unique():
                if cont % 100 == 0:
                    end = time.time()
                    print("\nusuário: ", cont)
                    print("\nduração: ", (end - init)/60)

                cont += 1
                users_checkins_sorted = users_checkins[users_checkins[userid_column] == user_id].sort_values(by=[datetime_column])
                locations = users_checkins_sorted[locationid_column].tolist()

                for i in range(len(locations)):
                    current_location = locationid_to_int[locations[i]]
                    for j in range(1, 6):
                        if ((i - j) < 0):
                            break
                        self.LL[current_location, locationid_to_int[locations[i - j]]] += 1
                    for j in range(1, 6):
                        if (i + j) > len(locations) - 1:
                            break
                        self.LL[current_location, locationid_to_int[locations[j + i]]] += 1

            init = time.time()
            end = time.time()
            print("\ncalculou os totais", (end - init)/60)
        except Exception as e:
            raise e

    def _create_LT_matrix(self, users_checinks, locationid_column, datetime_column, locationid_to_int):

        locations = users_checinks[locationid_column].tolist()
        datetimes = users_checinks[datetime_column].tolist()
        unique_locationsids = users_checinks[locationid_column].unique().tolist()
        total_locations = len(unique_locationsids)
        Dt = np.zeros((total_locations, 48))

        for i in range(len(locations)):
            current_location = locationid_to_int[locations[i]]
            if (datetimes[i].weekday() >= Weekday.SATURDAY.value):
                Dt[current_location][datetimes[i].hour + 24] += 1
            else:
                Dt[current_location][datetimes[i].hour] += 1

        self.LT = Dt

    def define_pmi_matrices(self, users_checkins, userid_column, datetime_column, locationid_column, max_time_between_records):

        print("\nmatriz pmi")
        print("\n", users_checkins)
        print("\n", datetime_column)
        n_max_pois = 141900
        datetimes = users_checkins[datetime_column].tolist()
        location_ids = users_checkins[locationid_column].tolist()
        unique_location_ids = users_checkins[locationid_column].unique().tolist()
        ids = users_checkins[userid_column].unique().tolist()

        location_ids_to_matrix_index = {unique_location_ids[i]: i for i in range(len(unique_location_ids))}
        users_pois_ids = {i: [] for i in ids}
        location_time_pmi_matrix = dok_matrix((n_max_pois, 48), dtype=np.float32)

        count = 0
        max_timedelta = pd.Timedelta(days=2020)
        if len(max_time_between_records) > 0:
            max_timedelta = pd.Timedelta(days=int(max_time_between_records))
        for j in range(1, len(datetimes)):
            anterior = j - 1
            atual = j
            local_anterior = location_ids[anterior]
            local_atual = location_ids[atual]
            # retirar eventos muito esparços
            if len(max_time_between_records) > 0:
                if (datetimes[atual] - datetimes[anterior]) > max_timedelta:
                    continue
            # retirar eventos consecutivos em um mesmo estabelecimento

            if local_anterior == local_atual:
                continue

            current_datetime = datetimes[atual]
            hour = current_datetime.hour
            if current_datetime.weekday() > 4:
                hour = hour + 24
            location_time_pmi_matrix[location_ids_to_matrix_index[local_atual], hour] += 1

        print("\nL T")
        print("\n", location_time_pmi_matrix)

        total_hour = [sum(location_time_pmi_matrix[:, i]) for i in range(len(location_time_pmi_matrix[0]))]

        for i in range(len(location_time_pmi_matrix)):

            for j in range(len(location_time_pmi_matrix[i])):
                pass

    def delete_files(self, files):

        for file in files:
            if os.path.exists(file):
                os.remove(file)

    def generate_pattern_matrices(self,
                                  users_checkin,
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
                                  location_location_pmi_matrix_filename,
                                  location_time_omi_matrix_filename,
                                  int_to_locationid_filename,
                                  userid_column,
                                  category_column,
                                  locationid_column,
                                  latitude_column,
                                  longitude_column,
                                  datetime_column,
                                  differemt_venues,
                                  directed,
                                  personal_features_matrix,
                                  top_users,
                                  max_time_between_records,
                                  num_users,
                                  base,
                                  hour48=True,
                                  osm_category_column=None):

        if osm_category_column is not None:
            category_column = osm_category_column

        # shuffle
        users_checkin = users_checkin.sample(frac=1, random_state=1).reset_index(drop=True)
        print("\n", users_checkin)
        users_checkin = users_checkin.dropna(subset=[userid_column, category_column, locationid_column, datetime_column])
        users_checkin = users_checkin.query(category_column + " != ''")
        new_ids = []
        adj_matrices_column = []
        feat_matrices_column = []
        sequence_matrices_column = []
        categories_column = []
        count = 0
        num_users = 20000

        files_names = [adjacency_matrix_filename,
                       adjacency_weekday_matrix_filename,
                       adjacency_weekend_matrix_filename,
                       temporal_matrix_filename,
                       temporal_weekday_matrix_filename,
                       temporal_weekend_matrix_filename,
                       distance_matrix_filename,
                       duration_matrix_filename
                       ]
        files_names = [i.replace("8_c", "7_c") for i in files_names]
        self.delete_files(files_names + [location_time_omi_matrix_filename, locationid_column, location_location_pmi_matrix_filename, int_to_locationid_filename])
        start = time.time()
        users_checkin[userid_column] = users_checkin[userid_column].to_numpy()
        users_checkin[locationid_column] = users_checkin[locationid_column].astype(int)
        original_columns = users_checkin.columns.tolist()
        unique_locationsids = users_checkin[locationid_column].unique().tolist()
        locationid_to_int = {unique_locationsids[i]: i for i in range(len(unique_locationsids))}
        keys = list(locationid_to_int.keys())
        values = list(locationid_to_int.values())
        int_to_location_id = {values[i]: keys[i] for i in range(len(keys))}
        self._create_LT_matrix(users_checkin, locationid_column, datetime_column, locationid_to_int)
        print("\nterminou LT")
        lt = pd.DataFrame(self.LT, columns=[str(i) for i in range(self.LT.shape[1])])
        self.matrix_generation_for_poi_categorization_loader.save_df_to_csv(lt, location_time_omi_matrix_filename.replace("8_cat", "7_cat"))
        pd.DataFrame({'locationid': keys, 'int': values}).to_csv(int_to_locationid_filename.replace("8_cat", "7_cat"), index=False)
        lt = ""
        self.LT = ""
        self._create_location_coocurrency_matrix(users_checkin, userid_column, datetime_column, locationid_column, locationid_to_int)
        print("\nterminou LL")
        self.matrix_generation_for_poi_categorization_loader.save_sparse_matrix_to_npz(sparse.csr_matrix(self.LL), location_location_pmi_matrix_filename.replace("8_c", "7_c"))
        self.LL = ""
        users_checkin_0 = ""


        users_checkin = users_checkin.groupby(userid_column).apply(lambda e: self.generate_user_matrices(e, e[userid_column].iloc[0],
                                                                                                         datetime_column,
                                                                                                         locationid_column,
                                                                                                         category_column,
                                                                                                         latitude_column,
                                                                                                         longitude_column,
                                                                                                         osm_category_column,
                                                                                                         dataset_name,
                                                                                                         categories_type,
                                                                                                         base,
                                                                                                         files_names,
                                                                                                         differemt_venues, personal_features_matrix, hour48, directed, max_time_between_records))
        print("\nFIM")
        end = time.time()
        print("\nDuração: ", (end - start)/60)

    def generate_gpr_matrices_v2(self,
                                  users_checkin,
                                  dataset_name,
                                  adjacency_matrix_filename,
                                  distance_matrix_filename,
                                 user_poi_matrix_filename,
                                  userid_column,
                                  category_column,
                                  locationid_column,
                                  latitude_column,
                                  longitude_column,
                                  datetime_column):

        # shuffle
        users_checkin = users_checkin.sample(frac=1, random_state=1).reset_index(drop=True)
        del users_checkin['Unnamed: 0']
        users_checkin = users_checkin.dropna(subset=[userid_column, category_column, locationid_column, datetime_column])
        users_checkin = users_checkin.query(category_column + " != ''")
        new_ids = []
        adj_matrices_column = []
        feat_matrices_column = []
        sequence_matrices_column = []
        categories_column = []
        count = 0
        num_users = 15000

        files_names = [adjacency_matrix_filename,
                       distance_matrix_filename,
                       user_poi_matrix_filename
                       ]
        self.delete_files(
            files_names)
        ids = users_checkin[userid_column].unique().tolist()
        users_checkin = users_checkin[users_checkin[userid_column].isin(ids[:num_users])]
        start = time.time()
        users_checkin[userid_column] = users_checkin[userid_column].to_numpy()
        users_checkin[locationid_column] = users_checkin[locationid_column].astype(int)
        original_columns = users_checkin.columns.tolist()

        users_checkin = users_checkin.groupby(userid_column).apply(lambda e: self.generate_gpr_user_matrices(e, e[userid_column].iloc[0],
                                                                                                         datetime_column,
                                                                                                         locationid_column,
                                                                                                         category_column,
                                                                                                         latitude_column,
                                                                                                         longitude_column,
                                                                                                         dataset_name,
                                                                                                         files_names))
        end = time.time()

    def generate_gpr_matrices(self,
                                  users_checkin,
                                  adjacency_matrix_filename,
                                  features_matrix_filename,
                                  userid_column,
                                    latitude_column,
                                    longitude_column,
                                  category_column,
                                  locationid_column,
                                  datetime_column,
                                  directed,
                                  osm_category_column=None):

        print("\npontos", osm_category_column)
        # constantes
        tempo_limite = 6

        if osm_category_column is not None:
            category_column = osm_category_column
        users_checkin = users_checkin
        users_checkin[datetime_column] = pd.to_datetime(users_checkin[datetime_column],
                                                        infer_datetime_format=True)

        users_checkin = users_checkin.dropna(
            subset=[userid_column, category_column, locationid_column, datetime_column])
        # shuffle
        users_checkin = users_checkin.sample(frac=1).reset_index(drop=True)
        ids = users_checkin[userid_column].unique().tolist()
        new_ids = []
        adj_matrices_column = []
        feat_matrices_column = []
        user_poi_vector_column = []
        categories_column = []
        count = 0
        it = 0
        for id_ in ids:
            it += 1
            query = userid_column + "==" + "'" + str(id_) + "'"
            user_checkin = users_checkin.query(query)

            user_checkin = user_checkin.sort_values(by=[datetime_column])

            n_pois = len(user_checkin[locationid_column].unique().tolist())
            poi_poi_graph = [[0 for i in range(n_pois)] for j in range(n_pois)]
            features_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
            # user-poi graph (vetor que contabiliza a quantidade de visitas em cada POI)
            user_poi_vector = [0] * n_pois
            categories_list = user_checkin[category_column].tolist()

            datetimes = user_checkin[datetime_column].tolist()
            placeids = user_checkin[locationid_column].tolist()
            latitudes = user_checkin[latitude_column].tolist()
            longitudes = user_checkin[longitude_column].tolist()
            placeids_unique = user_checkin[locationid_column].unique().tolist()
            placeids_unique_to_int = {placeids_unique[i]: i for i in range(len(placeids_unique))}
            # converter os ids dos locais para inteiro
            placeids_int = [placeids_unique_to_int[placeids[i]] for i in range(len(placeids))]
            categories = user_checkin[category_column].tolist()
            if len(categories) < 2:
                continue

            # inicializar

            categories_list[0] = categories[0]
            user_poi_vector[placeids_int[0]] = 1

            for j in range(1, len(datetimes)):
                anterior = j - 1
                atual = j
                poi_anterior = placeids[anterior]
                poi_atual = placeids[atual]

                if (datetimes[atual] - datetimes[anterior]).total_seconds()/360 > tempo_limite:
                    continue

                if categories[atual] == "":
                    continue
                if directed:
                    poi_poi_graph[placeids_int[anterior]][placeids_int[atual]] += 1
                else:
                    poi_poi_graph[placeids_int[anterior]][placeids_int[atual]] += 1
                    poi_poi_graph[placeids_int[atual]][placeids_int[anterior]] += 1

                categories_list[placeids_int[atual]] = categories[atual]

                user_poi_vector[placeids_int[atual]] += 1

                # feature matrix (calcular distancia)
                if features_matrix[placeids_int[anterior]][placeids_int[atual]] != 0:
                    continue
                if features_matrix[placeids_int[anterior]][placeids_int[atual]] == 0 \
                        and placeids_int[anterior] != placeids_int[atual]:

                    di = points_distance([latitudes[anterior],
                                        longitudes[anterior]],
                                        [latitudes[atual],
                                        longitudes[atual]])

                    features_matrix[placeids_int[anterior]][placeids_int[atual]] = di
                    features_matrix[placeids_int[atual]][placeids_int[anterior]] = di

            poi_poi_graph, features_matrix, categories_list = self.remove_gpr_pois_that_dont_have_categories(
                categories_list, poi_poi_graph, features_matrix)
            if poi_poi_graph != []:
                count += 1

            if len(poi_poi_graph) <= 2 or len(features_matrix) <= 2 or len(categories_list) <= 2:
                continue

            new_ids.append(i)
            adj_matrices_column.append(str(poi_poi_graph))
            feat_matrices_column.append(str(features_matrix))
            user_poi_vector_column.append(str(user_poi_vector))
            categories_column.append(categories_list)

        print("\nFiltro: ", count)
        print("\ntamanhos: ", len(new_ids), len(adj_matrices_column), len(categories_column))
        adjacency_matrix_df = pd.DataFrame(data={"user_id": new_ids,
                                                 "matrices": adj_matrices_column,
                                                 "category": categories_column})

        features_matrix_df = pd.DataFrame(data={"user_id": new_ids,
                                                "matrices": feat_matrices_column,
                                                "category": categories_column})

        user_poi_matrix_df = pd.DataFrame(data={"user_id": new_ids,
                                                "vector": user_poi_vector_column})

        print("\n", adjacency_matrix_df)
        adjacency_matrix_filename = adjacency_matrix_filename.replace("matrizes", "gpr")
        features_matrix_filename = features_matrix_filename.replace("matrizes", "gpr")

        self.matrix_generation_for_poi_categorization_loader. \
            save_df_to_csv(adjacency_matrix_df, adjacency_matrix_filename)

        self.matrix_generation_for_poi_categorization_loader. \
            save_df_to_csv(features_matrix_df, features_matrix_filename)

        self.matrix_generation_for_poi_categorization_loader.\
            save_df_to_csv(user_poi_matrix_df, adjacency_matrix_filename.replace("adjacency_matrix", "user_poi_vector"))

    def _distance_between_pois(self, users_checkin, locationid_column,
                               latitude_column, longitude_column, n_pois):

        features_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        pois = users_checkin.groupby(locationid_column).apply(lambda e: e[[latitude_column, longitude_column]].iloc[0]).reset_index()


    def categories_preproccessing(self, categories, categories_to_int_osm):

        c = []
        for i in range(len(categories)):
            cate = categories_to_int_osm[categories[i].split(":")[0]]
            c.append(cate)

        return c

    def remove_gps_pois_that_dont_have_categories(self, categories, adjacency_matrix, features_matrix):

        indexes_filtered_pois = []
        adjacency_matrix = np.array(adjacency_matrix)
        features_matrix = np.array(features_matrix)
        for i in range(len(categories)):
            if categories[i] >= 0:
                indexes_filtered_pois.append(i)

        indexes_filtered_pois = np.array(indexes_filtered_pois)
        if len(indexes_filtered_pois) <= 1:
            return [], [], []

        categories = np.array(categories)
        categories = categories[indexes_filtered_pois]
        adjacency_matrix = adjacency_matrix[indexes_filtered_pois[:, None], indexes_filtered_pois]
        features_matrix = features_matrix[indexes_filtered_pois, :]

        if len(adjacency_matrix) <= 1:
            adjacency_matrix = []
            features_matrix = []

        return adjacency_matrix.tolist(), features_matrix.tolist(), categories.tolist()

    def remove_gpr_gps_pois_that_dont_have_categories(self, categories, adjacency_matrix, distance_matrix):

        indexes_filtered_pois = []
        adjacency_matrix = np.array(adjacency_matrix)
        distance_matrix = np.array(distance_matrix)
        for i in range(len(categories)):
            if categories[i] >= 0:
                indexes_filtered_pois.append(i)

        indexes_filtered_pois = np.array(indexes_filtered_pois)
        if len(indexes_filtered_pois) <= 1:
            return [], [], [], []

        categories = np.array(categories)
        categories = categories[indexes_filtered_pois]
        adjacency_matrix = adjacency_matrix[indexes_filtered_pois[:, None], indexes_filtered_pois]
        distance_matrix = distance_matrix[indexes_filtered_pois[:, None], indexes_filtered_pois]

        if len(adjacency_matrix) <= 1:
            adjacency_matrix = []
            distance_matrix = []
            user_poi_matrix = []

        return adjacency_matrix.tolist(), distance_matrix.tolist(), categories.tolist()


    def remove_raw_gps_pois_that_dont_have_categories(self, categories, adjacency_matrix, features_matrix):

        indexes_filtered_pois = []
        adjacency_matrix = np.array(adjacency_matrix)
        features_matrix = np.array(features_matrix)
        for i in range(len(categories)):
            if len(categories[i]) >= 0 and categories[i][0] != "":
                indexes_filtered_pois.append(i)

        indexes_filtered_pois = np.array(indexes_filtered_pois)
        if len(indexes_filtered_pois) <= 1:
            return [], [], []

        categories = np.array(categories)
        categories = categories[indexes_filtered_pois]
        adjacency_matrix = adjacency_matrix[indexes_filtered_pois[:, None], indexes_filtered_pois]
        features_matrix = features_matrix[indexes_filtered_pois, :]

        if len(adjacency_matrix) <= 1:
            adjacency_matrix = []
            features_matrix = []

        return adjacency_matrix.tolist(), features_matrix.tolist(), categories.tolist()

    def remove_gpr_pois_that_dont_have_categories(self,
                                                  categories,
                                                  adjacency_matrix,
                                                  features_matrix):

        indexes_filtered_pois = []
        adjacency_matrix = np.array(adjacency_matrix)
        features_matrix = np.array(features_matrix)
        for i in range(len(categories)):
            if categories[i] >= 0:
                indexes_filtered_pois.append(i)

        indexes_filtered_pois = np.array(indexes_filtered_pois)
        if len(indexes_filtered_pois) <= 1:
            return [], [], []

        categories = np.array(categories)
        categories = categories[indexes_filtered_pois]
        adjacency_matrix = adjacency_matrix[indexes_filtered_pois[:, None], indexes_filtered_pois]
        features_matrix = features_matrix[indexes_filtered_pois, :]

        if len(adjacency_matrix) <= 1:
            adjacency_matrix = []
            features_matrix = []

        return adjacency_matrix.tolist(), features_matrix.tolist(), categories.tolist()

    def categories_list_preproccessing(self, categories, categories_to_int_osm):

        user_categories = []
        for i in range(len(categories)):
            c = []
            categories_names = categories[i].replace("'", "").replace(" ", "").replace("[", "").replace("]", "").split(",")
            print("\nelement", categories_names)
            for category in categories_names:
                print("\ncategorias: ", category)
                if category == "" or category == ' ':
                    c.append("")
                    continue
                cate = categories_to_int_osm[category]
                c.append(cate)
            user_categories.append(c)
        return user_categories

    def _summarize_categories_distance_matrix(self, categories_distances_matrix):
        sigma = 10
        categories_distances_list = []
        for row in range(len(categories_distances_matrix)):

            category_distances_list = []
            for column in range(len(categories_distances_matrix[row])):

                values = categories_distances_matrix[row][column]

                if len(values) == 0:
                    categories_distances_matrix[row][column] = 0
                    category_distances_list.append(0)
                else:

                    d_cc = st.median(values)
                    categories_distances_matrix[row][column] = d_cc

        return categories_distances_matrix

    def _duration_importance(self, duration):

        duration = duration * duration
        duration = -(duration / (self.duration_sigma * self.duration_sigma))
        duration = math.exp(duration)

        return duration

    def _distance_importance(self, distance):

        distance = distance * distance
        distance = -(distance / (self.distance_sigma * self.distance_sigma))
        distance = math.exp(distance)

        return distance