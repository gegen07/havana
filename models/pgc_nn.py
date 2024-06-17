import tensorflow as tf
from spektral.layers.convolutional import ARMAConv
from tensorflow.keras.layers import Input, Dense,  Dropout
from tensorflow.keras.models import Model

class PGC_NN:

    def __init__(self, params):

        self.max_size_matrices = params["max_size_matrices"]
        self.max_size_sequence = params["max_size_sequence"]
        self.num_classes = params["num_classes"]
        self.features_num_columns = params["features_num_columns"]
        self.share_weights = params["share_weights"]
        self.dropout_skip = params["dropout_skip"]
        self.dropout = params["dropout"]


    def build(self, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)

        A_input = Input((self.max_size_matrices,self.max_size_matrices))
        A_week_input = Input((self.max_size_matrices, self.max_size_matrices))
        A_weekend_input =  Input((self.max_size_matrices, self.max_size_matrices))
        Temporal_input = Input((self.max_size_matrices, self.features_num_columns))
        Temporal_week_input = Input((self.max_size_matrices, 24))
        Temporal_weekend_input = Input((self.max_size_matrices, 24))
        Distance_input = Input((self.max_size_matrices,self.max_size_matrices))
        Duration_input = Input((self.max_size_matrices,self.max_size_matrices))
        A_week_input = Input((self.max_size_matrices, self.max_size_matrices))
        Location_time_input = Input((self.max_size_matrices, self.features_num_columns))
        Location_location_input = Input((self.max_size_matrices, self.max_size_matrices))

        out_temporal = ARMAConv(20, activation='elu',
                                gcn_activation='gelu', share_weights=self.share_weights,
                                dropout_rate=self.dropout_skip)([Temporal_input, A_input])
        out_temporal = Dropout(self.dropout)(out_temporal)
        out_temporal = ARMAConv(self.num_classes,
                                     activation="softmax")([out_temporal, A_input])

        out_week_temporal = ARMAConv(20, activation='elu',
                                     gcn_activation='gelu', share_weights=self.share_weights,
                                     dropout_rate=self.dropout_skip)([Temporal_week_input, A_week_input])
        out_week_temporal = Dropout(self.dropout)(out_week_temporal)
        out_week_temporal = ARMAConv(self.num_classes,
                                     activation="softmax")([out_week_temporal, A_week_input])

        out_weekend_temporal = ARMAConv(20, activation='elu',
                                        gcn_activation='gelu', share_weights=self.share_weights,
                                        dropout_rate=self.dropout_skip)([Temporal_weekend_input, A_weekend_input])
        out_weekend_temporal = Dropout(self.dropout)(out_weekend_temporal)
        out_weekend_temporal = ARMAConv(self.num_classes,
                                     activation="softmax")([out_weekend_temporal, A_weekend_input])

        out_distance = ARMAConv(20, activation='elu',
                                gcn_activation='gelu')([Distance_input, A_input])
        out_distance = Dropout(self.dropout)(out_distance)
        out_distance = ARMAConv(self.num_classes,
                                     activation="softmax")([out_distance, A_input])

        out_duration = ARMAConv(20, activation='elu',
                                gcn_activation='gelu')([Duration_input, A_input])
        out_duration = Dropout(self.dropout)(out_duration)
        out_duration = ARMAConv(self.num_classes,
                                     activation="softmax")([out_duration, A_input])


        # usa
        out_location_location = ARMAConv(20, activation='elu',
                                gcn_activation='gelu')([Location_time_input, Location_location_input])
        out_location_location = Dropout(self.dropout)(out_location_location)
        out_location_location = ARMAConv(self.num_classes,
                                     activation="softmax")([out_location_location, Location_location_input])

        # usa
        out_location_time = Dense(40, activation='relu')(Location_time_input)
        out_location_time = Dense(self.num_classes,
                                     activation="softmax")(out_location_time)

        out_dense = tf.Variable(2.) * out_location_location + tf.Variable(2.) * out_location_time
        out_dense = Dense(self.num_classes,
                                     activation="softmax")(out_dense)

        out_gnn = tf.Variable(1.) * out_temporal + tf.Variable(1.) * out_week_temporal + tf.Variable(1.) * out_weekend_temporal + tf.Variable(1.) * out_distance + tf.Variable(1.) * out_duration
        out_gnn = Dense(self.num_classes,
                                     activation="softmax")(out_gnn)
        out = tf.Variable(1.) * out_dense + tf.Variable(1.) * out_gnn



        model = Model(inputs=[A_input, A_week_input, A_weekend_input, Temporal_input, Temporal_week_input, Temporal_weekend_input, Distance_input, Duration_input, Location_time_input, Location_location_input], outputs=[out])

        return model
