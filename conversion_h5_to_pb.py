# ~ from tensorflow.python.framework.ops import disable_eager_execution
# ~ disable_eager_execution()

import math
import network_libs.helper_DeepPilot4Pose as helper_net
import network_libs.DeepPilot4Pose_net as net
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K 
from tensorflow.compat.v1 import graph_util
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

if __name__ == "__main__":
    # Test model
    model = net.create_DeepPilot4Pose()
    
    
    path = "models/"
    model_name = path + "Test_Warehouse_model10_100ep.h5"
    output_model = "Test_Warehouse_model10_100ep"
    
    model.load_weights(model_name)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0, clipvalue=1.5)
    model.compile(optimizer=adam, loss={'pose_x': net.euc_lossx,
                                        'pose_y': net.euc_lossy,
                                        'pose_z': net.euc_lossz,
                                        'EMz': net.euc_lossEMz})
    model.summary()
    
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=path, name=output_model+".pb", as_text=False)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=path, name=output_model+".pbtxt", as_text=True)
    print("======================================================================")
    print("                         Save pb successfully!!!!")
    print("======================================================================")
    print(" ")

    # serialize model to JSON
    json_model = model.to_json()
    with open(path+output_model+".json", "w") as json_file:
        json_file.write(json_model)
    
    print("======================================================================")
    print("                     Save JSON File successfully!!!!")
    print("======================================================================")
    print(" ")
