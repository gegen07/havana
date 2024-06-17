import tensorflow as tf
from spektral.layers.convolutional import GATConv
from tensorflow.keras.layers import Input, Dense,  Dropout, Concatenate, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class HAVANA_GAT:

    def __init__(self, params):

        self.max_size_matrices = params["max_size_matrices"]
        self.max_size_sequence = params["max_size_sequence"]
        self.num_classes = params["num_classes"]
        self.features_num_columns = params["features_num_columns"]
        self.share_weights = False
        self.dropout_skip = params["dropout_skip"]
        self.dropout = params["dropout"]
        self.num_pois = 3

    def build(self, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)

        l2_reg = l2(5e-4)  # L2 regularization rate
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


        out_temporal2 = GATConv(20, kernel_regularizer=l2_reg,
                                 share_weights=self.share_weights,
                                dropout_rate=self.dropout_skip)([Temporal_input, A_input])
        out_temporal2 = Dropout(self.dropout)(out_temporal2)
        out_temporal2 = GATConv(20, kernel_regularizer=l2_reg)([out_temporal2, A_input])

        out_week_temporal2 = GATConv(20, kernel_regularizer=l2_reg,
                                      share_weights=self.share_weights,
                                     dropout_rate=self.dropout_skip)([Temporal_week_input, A_week_input])
        out_week_temporal2 = Dropout(self.dropout)(out_week_temporal2)
        out_week_temporal2 = GATConv(20, kernel_regularizer=l2_reg)([out_week_temporal2, A_week_input])

        out_weekend_temporal2 = GATConv(20, kernel_regularizer=l2_reg,
                                         share_weights=self.share_weights,
                                        dropout_rate=self.dropout_skip)([Temporal_weekend_input, A_weekend_input])
        out_weekend_temporal2 = Dropout(self.dropout)(out_weekend_temporal2)
        out_weekend_temporal2 = GATConv(20, kernel_regularizer=l2_reg)([out_weekend_temporal2, A_weekend_input])

        out_distance2 = GATConv(20, kernel_regularizer=l2_reg,
                               )([Distance_input, A_input])
        out_distance2 = Dropout(self.dropout)(out_distance2)
        out_distance2 = GATConv(20, kernel_regularizer=l2_reg)([out_distance2, A_input])

        out_duration2 = GATConv(20, kernel_regularizer=l2_reg,
                               )([Duration_input, A_input])
        out_duration2 = Dropout(self.dropout)(out_duration2)
        out_duration2 = GATConv(20, kernel_regularizer=l2_reg)([out_duration2, A_input])


        # usa
        out_location_location2 = GATConv(20, kernel_regularizer=l2_reg,
                               )([Location_time_input, Location_location_input])
        out_location_location2 = Dropout(self.dropout)(out_location_location2)
        out_location_location2 = GATConv(20, kernel_regularizer=l2_reg)([out_location_location2, Location_location_input])

        # usa
        out_location_time2 = Dense(40, activation='relu')(Location_time_input)
        out_location_time2 = Dense(20, kernel_regularizer=l2_reg)(out_location_time2)

        out_dense2 = tf.Variable(2.) * out_location_location2 + tf.Variable(2.) * out_location_time2
        out_dense2 = Dense(20, kernel_regularizer=l2_reg)(out_dense2)

        omega_2 = tf.Variable(1.) * out_temporal2 + tf.Variable(1.) * out_week_temporal2 + tf.Variable(1.) * out_weekend_temporal2 + tf.Variable(1.) * out_distance2 + tf.Variable(1.) * out_duration2
        omega_2 = Dense(20, kernel_regularizer=l2_reg)(omega_2)
        omega_2 = tf.Variable(1.) * out_dense2 + tf.Variable(1.) * omega_2

        concat_ys_omega_2 = Concatenate()([out_temporal2, out_week_temporal2, out_weekend_temporal2, out_distance2, out_duration2, out_location_location2, out_location_time2, omega_2])



        c1 = concat_ys_omega_2
        att = Attention()([c1, c1])
        out = Concatenate()([c1, att])
        out = Dense(50,  activation='relu')(out)
        out = Dense(self.num_classes, activation='softmax')(out)




        model = Model(inputs=[A_input, A_week_input, A_weekend_input, Temporal_input, Temporal_week_input, Temporal_weekend_input, Distance_input, Duration_input, Location_time_input, Location_location_input], outputs=[out])

        return model

