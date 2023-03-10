import numpy as np
import onnx
import onnxruntime as rt

import os
import sys
import shutil
import platform
system_type = platform.system()
dlpath=os.environ['ESP_DL_PATH']
path =dlpath+ f'/tools/quantization_tool/{system_type.lower()}'
path2=dlpath+"/tools/quantization_tool"
path3=dlpath+"/tutorial/quantization_tool_example"
if system_type == 'Windows':
    path = path.replace('/', '\\')
    path2 = path2.replace('/', '\\')
    path3 = path3.replace('/', '\\')
sys.path.append(path)
sys.path.append(path2)

from optimizer import *
from calibrator import *
from evaluator import *

SoC="esp32"
Format="int8"
model_name="MNIST"
model_path = 'mnist_model_example.onnx'
pickle_file_path = 'mnist_calib.pickle'
pickle_test_data_path='mnist_test_data.pickle'

def fix_path_move(path,path2):
    if system_type == 'Windows':
        path = path.replace('/', '\\')
        path2 = path2.replace('/', '\\')
    shutil.move(path, path2)
def fix_path_copy(path,path2):
    if system_type == 'Windows':
        path = path.replace('/', '\\')
        path2 = path2.replace('/', '\\')
    shutil.copyfile(path, path2)

if Format=="int8":
    Format_type="<int8_t>"
    format_type="int8_t"
    Max=127
    Min=-128
else:
    Format_type="<int16_t>" 
    format_type="int16_t"
    Max=32767
    Min=-32768

if __name__ == "__main__":
    # Optimize the onnx model
    optimized_model_path = optimize_fp_model(model_path)

    # Calibration
    with open(pickle_test_data_path, 'rb') as f:
        (test_images, test_labels) = pickle.load(f)
    test_images = test_images / 255.0

    # Prepare the calibration dataset
    calib_dataset = test_images[0:5000:50]
    model_proto = onnx.load(optimized_model_path)

    print('Generating the quantization table:')
    calib = Calibrator(Format, 'per-tensor', 'minmax')
    calib.set_providers(['CPUExecutionProvider'])
    calib.generate_quantization_table(model_proto, calib_dataset, pickle_file_path)
    calib.export_coefficient_to_cpp(model_proto, pickle_file_path, SoC, '.', '{}_coefficient'.format(model_name.lower()), True)

    # Evaluate the performance
    layer=1
    for node in model_proto.graph.node:
        if node.name.startswith('fused'):
            print('L{}:{}'.format(layer,node.input[0][:-2]),'connect to L{}:'.format(layer+1),node.output[0])
            layer+=1
            

    print('\nEvaluating the performance on esp32s3:')
    eva = Evaluator(Format, 'per-tensor', 'esp32s3')
    eva.set_providers(['CPUExecutionProvider'])
    eva.generate_quantized_model(model_proto, pickle_file_path)

    output_names = [n.name for n in model_proto.graph.output]
    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(optimized_model_path, providers=providers)

    batch_size = 100
    batch_num = int(len(test_images) / batch_size)
    res = 0
    fp_res = 0
    input_name = m.get_inputs()[0].name

    print("L1>> Input shape:",m.get_inputs()[0].shape,'\nL{}>> Output shape: {}'.format(layer,m.get_outputs()[0].shape),'\nInfo: '+m.get_modelmeta().graph_name,m.get_modelmeta().graph_description,'\n')
    for i in range(batch_num):
        # int8_model
        [outputs, _] = eva.evalute_quantized_model(test_images[i * batch_size:(i + 1) * batch_size], False)
        res = res + sum(np.argmax(outputs[0], axis=1) == test_labels[i * batch_size:(i + 1) * batch_size])

        # floating-point model
        fp_outputs = m.run(output_names, {input_name: test_images[i * batch_size:(i + 1) * batch_size].astype(np.float32)})
        fp_res = fp_res + sum(np.argmax(fp_outputs[0], axis=1) == test_labels[i * batch_size:(i + 1) * batch_size])

    print('accuracy of {} model is: {}'.format(Format,(res / len(test_images))))
    print('accuracy of fp32 model is: %f' % (fp_res / len(test_images)))

    print('Creating {}_model.hpp'.format(model_name.lower()))
    s="#pragma once\n"
    s=s+"#include \"dl_layer_model.hpp\"\n#include \"dl_layer_reshape.hpp\"\n#include \"dl_layer_conv2d.hpp\"\n#include \"{}_coefficient.hpp\"\n#include <stdint.h>\n".format(model_name.lower())
    s=s+"\nusing namespace dl;\nusing namespace layer;\nusing namespace {}_coefficient;\n\n".format(model_name.lower())
    s=s+"class {} : public Model{}".format(model_name,Format_type)+'\n{\n'
    s=s+"private:\n"
    for i in range(layer+1):
        if i==0:
            continue
        elif i==1:
            s=s+"\tReshape{} l{};".format(Format_type,i)+'\n'
        elif i==layer:
            s=s+"\npublic:\n"
            s=s+"\tConv2D{} l{};".format(Format_type,i)+'\n\t'
        else:  
            s=s+"\tConv2D{} l{};".format(Format_type,i)+'\n'
    counter=0
    for i in range(layer+1):
        if i==0:
            continue
        elif i==1:
            s=s+"{}() :\tl1(Reshape{}({{1,1,{}}})),".format(model_name,Format_type,m.get_inputs()[0].shape[1]*m.get_inputs()[0].shape[2])+'\n\t\t\t\t'
        elif i==layer:
             s=s+"l{}(Conv2D{}(<fused_gemm_{}>, get_fused_gemm_{}_filter(), get_fused_gemm_{}_bias(), NULL, PADDING_SAME_END, {{}}, 1, 1, \"l{}\")){{}}".format(i,Format_type,i-2,counter,counter,i)+'\n\t'  
        else:    
            s=s+"l{}(Conv2D{}(<fused_gemm_{}>, get_fused_gemm_{}_filter(), get_fused_gemm_{}_bias(), get_fused_gemm_{}_activation(), PADDING_SAME_END, {{}}, 1, 1, \"l{}\")),".format(i,Format_type,i-2,counter,counter,counter,i)+'\n\t\t\t\t'
            counter+=1
    s=s+"void build(Tensor{} &input)".format(Format_type)+'\n\t{\n\t\t'   
    for i in range(layer+1):
        if i==0:
            continue
        elif i==1:
            s=s+"this->l1.build(input);"+"\n\t\t"  
        elif i==layer:
             s=s+"this->l{}.build(this->l{}.get_output());".format(i,i-1)+"\n\t}\n\t"            
        else:    
            s=s+"this->l{}.build(this->l{}.get_output());".format(i,i-1)+"\n\t\t"   
    s=s+"void call(Tensor{} &input)".format(Format_type)+'\n\t{\n\t\t'   
    for i in range(layer+1):
        if i==0:
            continue
        elif i==1:
            s=s+"this->l1.call(input);"+"\n\t\t" 
            s=s+"input.free_element();"+"\n\n\t\t" 
        elif i==layer:
            s=s+"this->l{}.call(this->l{}.get_output());".format(i,i-1)+"\n\t\t"   
            s=s+"this->l{}.get_output().free_element();".format(i-1)+"\n\t}\n"             
        else:    
            s=s+"this->l{}.call(this->l{}.get_output());".format(i,i-1)+"\n\t\t"   
            s=s+"this->l{}.get_output().free_element();".format(i-1)+"\n\n\t\t"        

    s=s+"\n};\n"
    with open("{}_model.hpp".format(model_name.lower()), "w") as f:
        f.write(s)

    print('Creating app_main.cpp')
    g="#include <stdio.h>\n"
    g=g+"#include <stdlib.h>\n\n"
    g=g+"#include \"esp_system.h\"\n"
    g=g+"#include \"freertos/FreeRTOS.h\"\n"
    g=g+"#include \"freertos/task.h\"\n\n"
    g=g+"#include \"dl_tool.hpp\"\n"
    g=g+"#include \"{}_model.hpp\"\n\n".format(model_name.lower())
    g=g+"extern \"C\" void app_main(void)\n{\n\t"
    g=g+"int input_height = {};\n\t".format(m.get_inputs()[0].shape[-1])
    g=g+"int input_width = {};\n\t".format(m.get_inputs()[0].shape[-2])
    g=g+"int input_channel = 1;\n\t"
    g=g+"int input_exponent = <output_exponent>;\n\t"
    g=g+"{} *model_input = ({} *)dl::tool::malloc_aligned_prefer(input_height*input_width*input_channel, sizeof({} *));\n\t".format(format_type,format_type,format_type)
    g=g+"for(int i=0 ;i<input_height*input_width*input_channel; i++){\n\t\t"
    g=g+"float normalized_input = test_image[i] / 255.0; //normalization\n\t\t"
    g=g+"model_input[i] = ({})DL_CLIP(normalized_input * (1 << -input_exponent), {}, {});\n\t}}\n\t".format(format_type,Min,Max)
    g=g+"Tensor{} input;\n\t".format(Format_type)
    g=g+"input.set_element(({} *)model_input).set_exponent(input_exponent).set_shape({{input_height, input_width, input_channel}}).set_auto_free(false);\n".format(format_type)
    g=g+"\n\t{} model;\n\t".format(model_name)
    g=g+"dl::tool::Latency latency;\n\tlatency.start();\n\t"
    g=g+"model.forward(input);\n\tlatency.end();\n\tlatency.print(\"{}\", \"forward\");\n\n\t".format(model_name)
    g=g+"//parse\n\t{} *score = model.l{}.get_output().get_element_ptr();\n\t".format(format_type,layer)
    g=g+"{} max_score = score[0];\n\tint max_index = 0;\n\tprintf(\"%d, \", max_score);\n\n\t".format(format_type)
    g=g+"for (size_t i = 1; i < {}; i++)\n\t{{\n\t\t".format(m.get_outputs()[0].shape[-1])
    g=g+"printf(\"%d, \", score[i]);\n\t\tif (score[i] > max_score)\n\t\t{\n\t\t\tmax_score = score[i];\n\t\t\tmax_index = i;\n\t\t}\n\t}\n\t"
    g=g+"printf(\"Prediction Result: %d\", max_index);\n}"
    with open("app_main.cpp", "w") as f:
        f.write(g)
        
    print("Sorting files...")  
    if not os.path.exists("main"):
        os.makedirs("main")   
    if not os.path.exists("model"):
        os.makedirs("model") 	
    src_path = "{}_model.hpp".format(model_name.lower())
    dst_path = "model/{}_model.hpp".format(model_name.lower())
    fix_path_move(src_path,dst_path)
    src_path = "{}_coefficient.cpp".format(model_name.lower())
    dst_path = "model/{}_coefficient.cpp".format(model_name.lower())
    fix_path_move(src_path,dst_path)
    src_path = "{}_coefficient.hpp".format(model_name.lower())
    dst_path = "model/{}_coefficient.hpp".format(model_name.lower())
    fix_path_move(src_path,dst_path)
    src_path = "app_main.cpp"
    dst_path = "main/app_main.cpp"
    fix_path_move(src_path,dst_path)
    # src_path = path3+'/main/CMakeLists.txt'
    # dst_path = "main/CMakeLists.txt"
    # fix_path_copy(src_path,dst_path)
    src_path = path3+'/CMakeLists.txt'
    dst_path = "CMakeLists.txt"
    fix_path_copy(src_path,dst_path)
    src_path = path3+'/sdkconfig.defaults'
    dst_path = "sdkconfig.defaults"
    fix_path_copy(src_path,dst_path)
    src_path = path3+'/sdkconfig.defaults.esp32'
    dst_path = "sdkconfig.defaults.esp32"
    fix_path_copy(src_path,dst_path)
    src_path = path3+'/sdkconfig.defaults.esp32s2'
    dst_path = "sdkconfig.defaults.esp32s2"
    fix_path_copy(src_path,dst_path)
    src_path = path3+'/sdkconfig.defaults.esp32s3'
    dst_path = "sdkconfig.defaults.esp32s3"
    fix_path_copy(src_path,dst_path)
    
    s="idf_build_get_property(target IDF_TARGET)\n\n"
    s=s+"set(srcs  app_main.cpp ../model/{}_coefficient.cpp)\n".format(model_name.lower())
    s=s+"set(src_dirs  ../model)\n"
    s=s+"set(include_dirs  ../model $ENV{ESP_DL_PATH}/include $ENV{ESP_DL_PATH}/include/tool $ENV{ESP_DL_PATH}/include/typedef $ENV{ESP_DL_PATH}/include/nn $ENV{ESP_DL_PATH}/include/layer)\n\n"
    s=s+"idf_component_register(SRCS ${srcs} SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs}\n"
    s=s+"REQUIRES ${requires})\n"
    s=s+"set(lib     libdl.a)\n\nif(${IDF_TARGET} STREQUAL \"esp32\")\n\t"
    s=s+"set(links   \"-L $ENV{ESP_DL_PATH}/lib/esp32\")\n\n"
    s=s+"elseif(${IDF_TARGET} STREQUAL \"esp32s2\")\n\tset(links   \"-L $ENV{ESP_DL_PATH}/lib/esp32s2\")\n\n"
    s=s+"elseif(${IDF_TARGET} STREQUAL \"esp32s3\")\n\tset(links   \"-L $ENV{ESP_DL_PATH}/lib/esp32s3\")\n\n"
    s=s+"endif()\n\ntarget_link_libraries(${COMPONENT_TARGET} ${links})\ntarget_link_libraries(${COMPONENT_TARGET} ${lib})"
    with open("main/CMakeLists.txt", "w") as f:
        f.write(s)