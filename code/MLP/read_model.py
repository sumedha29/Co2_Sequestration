import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.layers import Lambda


def MLP_single():
    input_1 = layers.Input(shape=(7,))
    x = layers.Dense(128, activation="relu")(input_1)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(1)(x)

    model = keras.Model(inputs=input_1,outputs=x)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(1e-4), metrics=['mae'])
    print(model.summary())
    return model

def MLP_multiple():
    input_1 = layers.Input(shape=(7,),name='first_input')
    x = layers.Dense(128, activation="relu")(input_1)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    out_1 = layers.Dense(1,name='gas')(x)

    x = layers.Dense(128, activation="relu")(input_1)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    out_2 = layers.Dense(1,name='pressure')(x)

    input_2 = layers.Input(shape=(3,),name='second_input')
    x = layers.Dense(128, activation="relu")(input_2)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    out_3 = layers.Dense(1,name='water')(x)

    model = tf.keras.Model(inputs=[input_1,input_2],outputs=[out_1,out_2,out_3])
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


def MLP_physics():
    bias=False
    input_1 = layers.Input(shape=(3,),name='first_input')
    time = layers.Input(shape=(1,),name='time')
    x_input = layers.Input(shape=(1,),name='x_input')
    y_input = layers.Input(shape=(1,),name='y_input')
    z_input = layers.Input(shape=(1,),name='z_input')
    perm_input = layers.Input(shape=(3,3),name='perm_input')

    input_int = layers.Concatenate(axis=1)([input_1,time,x_input,y_input,z_input])

    x = layers.Dense(128, activation="relu")(input_int)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    out_1 = layers.Dense(1,name='gas')(x)

    x = layers.Dense(128, activation="relu")(input_int)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    out_2 = layers.Dense(1,name='pressure')(x)

    input_2 = layers.Input(shape=(3,),name='second_input')
    x = layers.Dense(128, activation="relu")(input_2)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    out_3 = layers.Dense(1,name='water')(x)


    grad_out_1 = gradient_1(bias)([out_1, out_2, time, x_input, y_input, z_input, perm_input])
    grad_out_2 = gradient_2(bias)([out_1, out_2, time, x_input, y_input, z_input, perm_input])
    

    model = keras.Model(inputs=[input_1,time,x_input,y_input,z_input,perm_input,input_2],outputs=[out_1,out_2,out_3,grad_out_1,grad_out_2])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['mae'])
    return model