import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.layers import Lambda


def LSTM_single():
    input_1 = layers.Input(shape=(4,3),name='first_input')
    time = layers.Input(shape=(4,1),name='time')
    x_input = layers.Input(shape=(4,1),name='x_input')
    y_input = layers.Input(shape=(4,1),name='y_input')
    z_input = layers.Input(shape=(4,1),name='z_input')
    input_int = layers.Concatenate(axis=2)([input_1, time, x_input, y_input, z_input])
    lstm_1 = LSTM(units=128, activation='relu', return_sequences=True)(input_int) 
    lstm_2 = LSTM(units=64, activation='relu')(lstm_1) 
    hidden_1 = Dense(32, activation='relu')(lstm_2)
    hidden_2 = Dense(16, activation='relu')(hidden_1)
    hidden_3 = Dense(8, activation='relu')(hidden_2) 
    out1 = Dense(1, activation='relu', name='gas')(hidden_3)
    print('out1 done ~ ')

    model = tf.keras.Model([input_1,time,x_input,y_input,z_input],[out1])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['mae'])
    print(model.summary())
    return model

def LSTM_multiple():
    input_1 = layers.Input(shape=(4,3),name='first_input')
    prev_time = layers.Input(shape=(3,),name='prev_time')
    cur_time = layers.Input(shape=(1,),name='cur_time')
    time_temp = layers.Concatenate(axis=1)([prev_time, cur_time])
    time = layers.Reshape((4,1))(time_temp)
    prev_x_input = layers.Input(shape=(3,),name='prev_x_input')
    cur_x_input = layers.Input(shape=(1,),name='cur_x_input')
    x_input_temp = layers.Concatenate(axis=1)([prev_x_input, cur_x_input])
    x_input = layers.Reshape((4,1))(x_input_temp)
    prev_y_input = layers.Input(shape=(3,),name='prev_y_input')
    cur_y_input = layers.Input(shape=(1,),name='cur_y_input')
    y_input_temp = layers.Concatenate(axis=1)([prev_y_input, cur_y_input])
    y_input = layers.Reshape((4,1))(y_input_temp)
    prev_z_input = layers.Input(shape=(3,),name='prev_z_input')
    cur_z_input = layers.Input(shape=(1,),name='cur_z_input')
    z_input_temp = layers.Concatenate(axis=1)([prev_z_input, cur_z_input])
    z_input = layers.Reshape((4,1))(z_input_temp)

    input_int = layers.Concatenate(axis=2)([input_1, time, x_input, y_input, z_input])
    lstm_1 = LSTM(units=128, activation='relu', return_sequences=True)(input_int) 
    lstm_2 = LSTM(units=64, activation='relu')(lstm_1) 
    hidden_1 = Dense(32, activation='relu')(lstm_2)
    hidden_2 = Dense(16, activation='relu')(hidden_1)
    hidden_3 = Dense(8, activation='relu')(hidden_2) 
    out1 = Dense(1, activation='relu', name='gas')(hidden_3)
    print('out1 done ~')

    lstm_1 = LSTM(units=128, activation='relu', return_sequences=True)(input_int) 
    lstm_2 = LSTM(units=64, activation='relu')(lstm_1) 
    hidden_1 = Dense(32, activation='relu')(lstm_2)
    hidden_2 = Dense(16, activation='relu')(hidden_1)
    hidden_3 = Dense(8, activation='relu')(hidden_2) 
    out2 = Dense(1, activation='relu', name='pressure')(hidden_3)
    print('out2 done ~')

    input_2 = layers.Input(shape=(4, 3), name='second_input')
    lstm_1 = LSTM(units=128, activation='relu', return_sequences=True)(input_2) 
    lstm_2 = LSTM(units=64, activation='relu')(lstm_1) 
    hidden_1 = Dense(32, activation='relu')(lstm_2)
    hidden_2 = Dense(16, activation='relu')(hidden_1)
    hidden_3 = Dense(8, activation='relu')(hidden_2) 
    out3 = Dense(1, activation='relu', name='water')(hidden_3)
    print('out3 done ~')

    model = tf.keras.Model(inputs=[input_1,prev_time,cur_time,prev_x_input,cur_x_input,prev_y_input,cur_y_input,prev_z_input,cur_z_input,input_2],outputs=[out1,out2,out3])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['mae'])
    print(model.summary())
    return model


class gradient_1(tf.keras.layers.Layer):
    def __init__(self,bias=True):
        super(gradient_1, self).__init__()
        self.gas_scale = tf.keras.layers.Dense(1,use_bias=bias)
        self.pressure_scale = tf.keras.layers.Dense(1,use_bias=bias)
        self.time_scale = tf.keras.layers.Dense(1,use_bias=bias)
        self.perm_scale = tf.keras.layers.Dense(1,use_bias=bias)
        self.all_scale = tf.keras.layers.Dense(1,use_bias=bias)
        self.constant1 = self.add_weight("constant1")
        self.constant2 = self.add_weight("constant2")
        
        

    def call(self, params):
        out1, out2, cur_time, cur_x_input, cur_y_input, cur_z_input, perm_input = params 
        out1 = self.gas_scale(out1)
        out2 = self.pressure_scale(out2)
        cur_time = self.time_scale(cur_time)
        perm_input = self.perm_scale(perm_input)
        
        gradient_with_time = tf.keras.backend.gradients(out1,cur_time)[0]
        bias = tf.expand_dims(tf.convert_to_tensor([0.,0.,constant1]),0)
        bias = tf.expand_dims(bias,2)
        
        pressure_grad_x = tf.keras.backend.gradients(out2,cur_x_input)[0]
        pressure_grad_y = tf.keras.backend.gradients(out2,cur_y_input)[0]
        pressure_grad_z = tf.keras.backend.gradients(out2,cur_z_input)[0]
        
        pressure_grad = tf.convert_to_tensor([pressure_grad_x,pressure_grad_y,pressure_grad_z])
        pressure_grad = tf.keras.backend.permute_dimensions(pressure_grad,(1,0,2))
        coeff = (1-out1)/constant2
        
        m = tf.multiply(perm_input,(pressure_grad - bias))
        m_grad_x = tf.keras.backend.gradients(m,cur_x_input)[0]
        m_grad_y = tf.keras.backend.gradients(m,cur_y_input)[0]
        m_grad_z = tf.keras.backend.gradients(m,cur_z_input)[0]
        
        m_grad = m_grad_x + m_grad_y + m_grad_z
        m_final = tf.multiply(coeff, m_grad)
        eqn = -gradient_with_time - m_final
        eqn = self.all_scale(eqn)
        return eqn 
    

class gradient_2(tf.keras.layers.Layer):
    def __init__(self,bias=True):
        super(gradient_1, self).__init__()
        self.gas_scale = tf.keras.layers.Dense(1,use_bias=bias)
        self.pressure_scale = tf.keras.layers.Dense(1,use_bias=bias)
        self.time_scale = tf.keras.layers.Dense(1,use_bias=bias)
        self.perm_scale = tf.keras.layers.Dense(1,use_bias=bias)
        self.all_scale = tf.keras.layers.Dense(1,use_bias=bias)
        self.constant1 = self.add_weight("constant1")
        self.constant2 = self.add_weight("constant2")
        
        

    def call(self, params):
        out1, out2, cur_time, cur_x_input, cur_y_input, cur_z_input, perm_input = params 
        out1 = self.gas_scale(out1)
        out2 = self.pressure_scale(out2)
        cur_time = self.time_scale(cur_time)
        perm_input = self.perm_scale(perm_input)
        
        gradient_with_time = tf.keras.backend.gradients(out1,cur_time)[0]
        bias = tf.expand_dims(tf.convert_to_tensor([0.,0.,constant1]),0)
        bias = tf.expand_dims(bias,2)
        
        pressure_grad_x = tf.keras.backend.gradients(out2,cur_x_input)[0]
        pressure_grad_y = tf.keras.backend.gradients(out2,cur_y_input)[0]
        pressure_grad_z = tf.keras.backend.gradients(out2,cur_z_input)[0]
        
        pressure_grad = tf.convert_to_tensor([pressure_grad_x,pressure_grad_y,pressure_grad_z])
        pressure_grad = tf.keras.backend.permute_dimensions(pressure_grad,(1,0,2))
        coeff = out1/constant2
        
        m = tf.multiply(perm_input,(pressure_grad - bias))
        m_grad_x = tf.keras.backend.gradients(m,cur_x_input)[0]
        m_grad_y = tf.keras.backend.gradients(m,cur_y_input)[0]
        m_grad_z = tf.keras.backend.gradients(m,cur_z_input)[0]
        
        m_grad = m_grad_x + m_grad_y + m_grad_z
        m_final = tf.multiply(coeff, m_grad)
        eqn = gradient_with_time - m_final
        eqn = self.all_scale(eqn)
        return eqn

def LSTM_physics():
    input_1 = layers.Input(shape=(4,3),name='first_input')
    prev_time = layers.Input(shape=(3,),name='prev_time')
    cur_time = layers.Input(shape=(1,),name='cur_time')
    time_temp = layers.Concatenate(axis=1)([prev_time, cur_time])
    time = layers.Reshape((4,1))(time_temp)
    prev_x_input = layers.Input(shape=(3,),name='prev_x_input')
    cur_x_input = layers.Input(shape=(1,),name='cur_x_input')
    x_input_temp = layers.Concatenate(axis=1)([prev_x_input, cur_x_input])
    x_input = layers.Reshape((4,1))(x_input_temp)
    prev_y_input = layers.Input(shape=(3,),name='prev_y_input')
    cur_y_input = layers.Input(shape=(1,),name='cur_y_input')
    y_input_temp = layers.Concatenate(axis=1)([prev_y_input, cur_y_input])
    y_input = layers.Reshape((4,1))(y_input_temp)
    prev_z_input = layers.Input(shape=(3,),name='prev_z_input')
    cur_z_input = layers.Input(shape=(1,),name='cur_z_input')
    z_input_temp = layers.Concatenate(axis=1)([prev_z_input, cur_z_input])
    z_input = layers.Reshape((4,1))(z_input_temp)
    perm_input = layers.Input(shape=(3,3),name='perm_input')

    input_int = layers.Concatenate(axis=2)([input_1, time, x_input, y_input, z_input])
    lstm_1 = LSTM(units=128, activation='relu', return_sequences=True,unroll=True)(input_int) 
    lstm_2 = LSTM(units=64, activation='relu',unroll=True)(lstm_1) 
    hidden_1 = Dense(32, activation='relu')(lstm_2)
    hidden_2 = Dense(16, activation='relu')(hidden_1)
    hidden_3 = Dense(8, activation='relu')(hidden_2) 
    out1 = Dense(1, activation='relu', name='gas')(hidden_3)
    print('out1 done ~')

    lstm_1 = LSTM(units=128, activation='relu', return_sequences=True,unroll=True)(input_int) 
    lstm_2 = LSTM(units=64, activation='relu',unroll=True)(lstm_1) 
    hidden_1 = Dense(32, activation='relu')(lstm_2)
    hidden_2 = Dense(16, activation='relu')(hidden_1)
    hidden_3 = Dense(8, activation='relu')(hidden_2) 
    out2 = Dense(1, activation='relu', name='pressure')(hidden_3)
    print('out2 done ~')

    input_2 = layers.Input(shape=(4, 3), name='second_input')
    lstm_1 = LSTM(units=128, activation='relu', return_sequences=True,unroll=True)(input_2) 
    lstm_2 = LSTM(units=64, activation='relu',unroll=True)(lstm_1) 
    hidden_1 = Dense(32, activation='relu')(lstm_2)
    hidden_2 = Dense(16, activation='relu')(hidden_1)
    hidden_3 = Dense(8, activation='relu')(hidden_2) 
    out3 = Dense(1, activation='relu', name='water')(hidden_3)
    print('out3 done ~')

    
    def gradient_2(params):
        out1, out2, cur_time, cur_x_input, cur_y_input, cur_z_input, perm_input = params     
        gradient_with_time = tf.keras.backend.gradients(out1,cur_time)[0]
        bias = tf.expand_dims(tf.convert_to_tensor([0.,0.,tf.keras.backend.variable(0.)]),0)
        bias = tf.expand_dims(bias,2)
        pressure_grad_x = tf.keras.backend.gradients(out2,cur_x_input)[0]
        pressure_grad_y = tf.keras.backend.gradients(out2,cur_y_input)[0]
        pressure_grad_z = tf.keras.backend.gradients(out2,cur_z_input)[0]

        pressure_grad = tf.convert_to_tensor([pressure_grad_x,pressure_grad_y,pressure_grad_z])
        pressure_grad = tf.keras.backend.permute_dimensions(pressure_grad,(1,0,2))
        coeff = (out1)/tf.keras.backend.variable(1e-8)     
        m = tf.multiply(perm_input,(pressure_grad - bias))
        m_grad_x = tf.keras.backend.gradients(m,cur_x_input)[0]
        m_grad_y = tf.keras.backend.gradients(m,cur_y_input)[0]
        m_grad_z = tf.keras.backend.gradients(m,cur_z_input)[0]

        m_grad = m_grad_x + m_grad_y + m_grad_z
        m_final = tf.multiply(coeff, m_grad)
        eqn = gradient_with_time - m_final
        return eqn



    grad_out_1 = gradient_1(bias)([out_1, out_2, time, x_input, y_input, z_input, perm_input])
    grad_out_2 = gradient_2(bias)([out_1, out_2, time, x_input, y_input, z_input, perm_input])
    
    model = tf.keras.Model(inputs=[input_1,prev_time,cur_time,prev_x_input,cur_x_input,prev_y_input,cur_y_input,prev_z_input,cur_z_input,perm_input,input_2],outputs=[out1,out2,out3,grad_out_1,grad_out_2])

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(1e-5), metrics=['mae'])
    print(model.summary())
    return model
