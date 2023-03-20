import numpy as np
import onnx
import onnxruntime as rt

import itertools
import re
import os
import sys
import pickle
import shutil
import platform
from tqdm import tqdm
from preprocess import preprocess

SoC="esp32"
Format="int8"
model_name="HANDRECOGNATION"
model_path = 'handrecognition_model.onnx'
pickle_test_data_path='Test.pkl'
pickle_calibration_data_path='Calibration.pkl'
class_names=['Palm','I','fist','fist_moved','thumb','index','ok','palm_moved','c','down']

system_type = platform.system()
dlpath=os.environ['ESP_DL_PATH']
pickle_file_path = 'quantization_table.pkl'
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
        (test_images,test_labels)= pickle.load(f)
    with open(pickle_calibration_data_path, 'rb') as f:
        (calib_images,calib_labels)= pickle.load(f)

    print("Original shape:",test_images.shape)
    test_images=preprocess(test_images)
    calib_images=preprocess(calib_images)
    print("Final shape:",test_images.shape)

    # Prepare the calibration dataset
    model_proto = onnx.load(optimized_model_path)
    onnx.checker.check_model(model_proto)

    print('Generating the quantization table:')
    calib = Calibrator(Format, 'per-tensor', 'minmax') #'per-channel'  'per-tensor'   'entropy' 'minmax'
    calib.set_providers(['CPUExecutionProvider'])
    calib.generate_quantization_table(model_proto, calib_images, pickle_file_path)
    calib.export_coefficient_to_cpp(model_proto, pickle_file_path, SoC, '.', '{}_coefficient'.format(model_name.lower()), True)

    # Evaluate the performance
            
    print('\nEvaluating the performance on esp32s3:')
    eva = Evaluator(Format, 'per-tensor', 'esp32s3')
    eva.set_providers(['CPUExecutionProvider'])
    eva.generate_quantized_model(model_proto, pickle_file_path)

    output_names = [n.name for n in model_proto.graph.output]
    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(optimized_model_path, providers=providers)

    layer,gemcount=1,0
    layers_name=[]
    maxpols_info=[]
    reshape_info=[]
    conv_dense_info=[]
    for node,info in itertools.zip_longest(model_proto.graph.node,model_proto.graph.value_info):
        if node.op_type=="Transpose" or node.op_type=="Relu":
            continue
        else:
            print('L{}:{} {}'.format(layer,node.op_type,node.name))
            if node.op_type=='Conv':
                s=node.name.replace('/','_').lower()
                conv_dense_info.append(s)
                layers_name.append('Conv2D')
            elif node.op_type=='Gemm':
                conv_dense_info.append("fused_gemm_"+str(gemcount))
                layers_name.append('Conv2D')
                gemcount+=1
            elif node.op_type=='MaxPool':
                maxpols_info.append([node.attribute[0].ints,node.attribute[1].ints])
                layers_name.append('MaxPool2D')
            elif node.op_type=='Reshape':
                if layer==1:
                    reshape_info.append(m.get_inputs()[0].shape[1]*m.get_inputs()[0].shape[2]*m.get_inputs()[0].shape[3])
                else:
                    reshape_info.append(info.type.tensor_type.shape.dim[1].dim_value)
                layers_name.append('Reshape')
            elif node.op_type=='Softmax':
                layers_name.append('Softmax')    
            layer+=1

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
    s=s+"#include \"dl_layer_model.hpp\"\n#include \"dl_layer_reshape.hpp\"\n#include \"dl_layer_conv2d.hpp\"\n#include \"dl_layer_max_pool2d.hpp\"\n#include \"dl_layer_softmax.hpp\"\n#include \"{}_coefficient.hpp\"\n#include <stdint.h>\n".format(model_name.lower())
    s=s+"\nusing namespace dl;\nusing namespace layer;\nusing namespace {}_coefficient;\n\n".format(model_name.lower())
    s=s+"class {} : public Model{}".format(model_name,Format_type)+'\n{\n'
    s=s+"private:\n"
    for k,v in enumerate(layers_name):
        if k==(len(layers_name)-1):
            s=s+"\npublic:\n"
            s=s+"\t{}{} l{};".format(v,Format_type,k+1)+'\n\n\t'
        else:
            s=s+"\t{}{} l{};".format(v,Format_type,k+1)+'\n'
    count,count2,count3=0,0,0
    for k,v in enumerate(layers_name):
        k=k+1
        if k==1:
            s=s+"{} () :\n\t\t\t\t".format(model_name)
        if v=='Reshape':
            s=s+"l{}(Reshape{}({{1,1,{}}},\"l{}_reshape\")),".format(k,Format_type,reshape_info[count2],k)+'\n\t\t\t\t'
            count2+=1
        elif v=='Conv2D' and k==(len(layers_name)-1):
            s=s+"l{}(Conv2D{}(<output_exponent>, get_{}_filter(), get_{}_bias(), NULL, PADDING_SAME_END, {{}}, 1, 1, \"l{}\")),".format(k,Format_type,conv_dense_info[count3],conv_dense_info[count3],k)+'\n\t\t\t\t' 
            count3+=1
        elif v=='Conv2D' and k==(len(layers_name)):
            s=s+"l{}(Conv2D{}(<output_exponent>, get_{}_filter(), get_{}_bias(), NULL, PADDING_SAME_END, {{}}, 1, 1, \"l{}\")){{}}".format(k,Format_type,conv_dense_info[count3],conv_dense_info[count3],k)+'\n\t\t\t\t' 
            count3+=1
        elif v=='Conv2D':    
            s=s+"l{}(Conv2D{}(<output_exponent>, get_{}_filter(), get_{}_bias(), get_{}_activation(), PADDING_SAME_END, {{}}, 1, 1, \"l{}\")),".format(k,Format_type,conv_dense_info[count3],conv_dense_info[count3],conv_dense_info[count3],k)+'\n\t\t\t\t'
            count3+=1
        elif v=="MaxPool2D":
            s=s+"l{}(MaxPool2D{}({{{},{}}},PADDING_VALID,{{}}, {}, {}, \"l{}\")),\n\t\t\t\t".format(k,Format_type,maxpols_info[count][1][1],maxpols_info[count][1][0],maxpols_info[count][0][0],maxpols_info[count][0][1],k)
            count+=1
        elif v=="Softmax":
            s=s+"l{}(Softmax{}(<output_exponent>,\"l{}\")){{}}\n\t".format(k,Format_type,k)
        else:
            print('Unknown layer!')

    s=s+"void build(Tensor{} &input)".format(Format_type)+'\n\t{\n\t\t'   
    for i in range(layer):
        if i==0:
            continue
        elif i==1:
            s=s+"this->l1.build(input);"+"\n\t\t"  
        elif i==layer-1:
             s=s+"this->l{}.build(this->l{}.get_output());".format(i,i-1)+"\n\t}\n\t"            
        else:    
            s=s+"this->l{}.build(this->l{}.get_output());".format(i,i-1)+"\n\t\t"   
    s=s+"void call(Tensor{} &input)".format(Format_type)+'\n\t{\n\t\t'   
    for i in range(layer):
        if i==0:
            continue
        elif i==1:
            s=s+"this->l1.call(input);"+"\n\t\t" 
            s=s+"input.free_element();"+"\n\n\t\t" 
        elif i==layer-1:
            s=s+"this->l{}.call(this->l{}.get_output());".format(i,i-1)+"\n\t\t"   
            s=s+"this->l{}.get_output().free_element();".format(i-1)+"\n\t}\n"             
        else:    
            s=s+"this->l{}.call(this->l{}.get_output());".format(i,i-1)+"\n\t\t"   
            s=s+"this->l{}.get_output().free_element();".format(i-1)+"\n\n\t\t"        

    s=s+"\n};\n"
    with open("{}_model.hpp".format(model_name.lower()), "w") as f:
        f.write(s)

    print('Creating inference.cpp')
    g="#include <stdio.h>\n"
    g=g+"#include <stdlib.h>\n\n"
    g=g+"#include \"esp_system.h\"\n"
    g=g+"#include \"freertos/FreeRTOS.h\"\n"
    g=g+"#include \"freertos/task.h\"\n"
    g=g+"#include <esp_log.h>\n\n"
    g=g+"#include \"dl_tool.hpp\"\n"
    g=g+"#include \"{}_model.hpp\"\n".format(model_name.lower())
    g=g+"#include \"esp_main.h\"\n\n"
    g=g+"int input_height = {};\n".format(m.get_inputs()[0].shape[-2])
    g=g+"int input_width = {};\n".format(m.get_inputs()[0].shape[-3])
    g=g+"int input_channel = 1;\n"
    g=g+"int input_exponent = <input_exponent>;\n"
    g=g+"int size=input_height*input_width*input_channel;\n"
    g=g+"static const char *TAG = \"INF\";\n\n"
    g=g+"#ifdef __cplusplus\n"
    g=g+"extern \"C\" {\n"
    g=g+"#endif\n\n"
    g=g+"int run_inference(void *image)\n{\n\t"
    g=g+"#ifdef CONFIG_PREPROCESS\n\t"
    g=g+"int8_t *pointer_to_img;\n\tpointer_to_img = (int8_t *) image;\n\t"
    g=g+"{} *model_input = ({} *)dl::tool::malloc_aligned_prefer(size, sizeof({} *));\n\t".format(format_type,format_type,format_type)
    g=g+"for(int i=0 ;i<size; i++){\n\t\t"
    g=g+"static float normalized_input = pointer_to_img[i] / 255.0; //normalization\n\t\t"
    g=g+"model_input[i] = ({})DL_CLIP(normalized_input * (1 << -input_exponent), {}, {});\n\t}}\n\t".format(format_type,Min,Max)
    g=g+"Tensor{} input;\n\t".format(Format_type)
    g=g+"input.set_element(({} *)model_input).set_exponent(input_exponent).set_shape({{input_height, input_width, input_channel}}).set_auto_free(false);\n".format(format_type)
    g=g+"\t#else\n\t"
    g=g+"Tensor{} input;\n\t".format(Format_type)
    g=g+"input.set_element(({} *)image).set_exponent(input_exponent).set_shape({{input_height, input_width, input_channel}}).set_auto_free(false);\n".format(format_type)
    g=g+"\t#endif\n\n\t"
    g=g+"{} model;\n\t".format(model_name)
    g=g+"#ifdef CONFIG_INFERENCE_LOG\n\t"
    g=g+"dl::tool::Latency latency;\n\tlatency.start();\n\t"
    g=g+"#endif\n\t"
    g=g+"model.forward(input);\n\t"
    g=g+"#ifdef CONFIG_INFERENCE_LOG\n\t"
    g=g+"latency.end();\n\tlatency.print(\"{}\", \"forward\");\n\n\t".format(model_name)
    g=g+"#endif\n\t#ifdef CONFIG_PREPROCESS\n\tdl::tool::free_aligned_prefer(model_input);\n\t#endif\n\n\t"
    g=g+"//parse\n\tauto *score = model.l{}.get_output().get_element_ptr();\n\t".format(layer-1)
    g=g+"auto max_score = score[0];\n\tint max_index = 0;\n\n\t"
    g=g+"for (size_t i = 1; i < {}; i++)\n\t{{\n\t\t".format(m.get_outputs()[0].shape[-1])
    g=g+"if (score[i] > max_score)\n\t\t{\n\t\t\tmax_score = score[i];\n\t\t\tmax_index = i;\n\t\t}\n\t}\n\t"
    g=g+"#ifdef CONFIG_INFERENCE_LOG\n\t"
    g=g+"switch (max_index)\n\t{\n\t"
    for i in range(m.get_outputs()[0].shape[-1]):
        g=g+"case {}:\n\t\tESP_LOGI(TAG,\"{}\")\n\t\tbreak;\n\t".format(i,class_names[i])
    g=g+"default:\n\t\tESP_LOGE(TAG,\"No result\")\n\t}\n\t#endif\n\treturn (max_index);\n}\n"
    g=g+"#ifdef __cplusplus\n}\n#endif"

    with open("inference.cpp", "w") as f:
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
    src_path = "inference.cpp"
    dst_path = "main/inference.cpp"
    fix_path_move(src_path,dst_path)
    src_path = path3+'/CMakeLists.txt'
    dst_path = "CMakeLists.txt"
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
    s=s+"set(CMAKE_CXX_STANDARD 17)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)\n\n"
    s=s+"set(srcs  main.c ../model/{}_coefficient.cpp inference.cpp esp_cli.c sd_card.c)\n".format(model_name.lower())
    s=s+"set(src_dirs  ../model)\n"
    s=s+"set(include_dirs  ../model ./include $ENV{ESP_DL_PATH}/include $ENV{ESP_DL_PATH}/include/tool $ENV{ESP_DL_PATH}/include/typedef $ENV{ESP_DL_PATH}/include/nn $ENV{ESP_DL_PATH}/include/layer $ENV{ESP_DL_PATH}/include/math)\n\n"
    s=s+"idf_component_register(SRCS ${srcs} SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs}\n"
    s=s+"REQUIRES ${requires})\n"
    s=s+"set(lib     libdl.a)\n\nif(${IDF_TARGET} STREQUAL \"esp32\")\n\t"
    s=s+"set(links   \"-L $ENV{ESP_DL_PATH}/lib/esp32\")\n\n"
    s=s+"elseif(${IDF_TARGET} STREQUAL \"esp32s2\")\n\tset(links   \"-L $ENV{ESP_DL_PATH}/lib/esp32s2\")\n\n"
    s=s+"elseif(${IDF_TARGET} STREQUAL \"esp32s3\")\n\tset(links   \"-L $ENV{ESP_DL_PATH}/lib/esp32s3\")\n\n"
    s=s+"endif()\n\ntarget_link_libraries(${COMPONENT_TARGET} ${links})\ntarget_link_libraries(${COMPONENT_TARGET} ${lib})"
    with open("main/CMakeLists.txt", "w") as f:
        f.write(s)

    print("Extracting test images...")
    f=open(pickle_test_data_path,'rb')
    data=pickle.load(f)
    f.close()
    X,Y=data
    print("Number of instances:",len(Y),'shape:',X[0].shape)
    destination="put_my_content_on_sdcard"
    if os.path.exists(destination):
        pass
    else:
        os.makedirs(destination)
    count=0
    for img in tqdm(X):
        img=preprocess(img)
        if(Format=="int8"):
            img = np.array(img, dtype=np.uint8)
        else:
            img = np.array(img, dtype=np.uint16)
        img.tofile(destination+"/"+str(count)+'-'+str(Y[count]))
        count=count+1

    print("Final shape: {}\nPlace exponent values in inference.cpp and {}_model.hpp".format(img.shape,model_name.lower()))