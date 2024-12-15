import re
import os
import json
import shutil
import numpy as np

class TreeLUTClassifier:
    def __init__(self, xgb_model, w_feature, w_tree, pipeline=[0, 0, 0], dir_path="./"):
        self._folder_created = False
        self._xgb_model = xgb_model
        self._objective = xgb_model.objective
        self._features = xgb_model.get_booster().feature_names
        if(self._features is None):
            self._features = [f"f{i}" for i in range(xgb_model.n_features_in_)]
        self._w_feature = w_feature
        self._w_tree = w_tree
        self._pipeline = pipeline
        self._dir_path = os.path.join(dir_path, 'TreeLUT')
        self._base_score = None
        self._threshold = None
        self._status = 'init'
        self._treelut_model = None
        self._n_features = xgb_model.n_features_in_
        self._n_trees_per_class = xgb_model.n_estimators
        self._sum_bit_length = None
        self._classes_bias = None
        if(self._objective == 'binary:logistic'):
            self._n_classes = 1
            config_dict = json.loads(xgb_model.get_booster().save_config())
            base_p = float(config_dict['learner']['learner_model_param']['base_score'])
            self._base_score = np.log(base_p / (1 - base_p))
        elif(self._objective == 'multi:softmax'):
            self._n_classes = xgb_model.n_classes_
        else:
            self._status = 'error'
            print("Error: TreeLUT currently supports the following XGBoost objectives: 'binary:logistic', 'multi:softmax'!")
        
    def convert(self):
        if(self._status == 'init'):
            self._format_model()
            self._quantize_model()
            self._status = 'quantized'
        elif(self._status == 'quantized'):
            print('Info: The model has already been converted!')
        else:
            print('Error!')

    def predict(self, X_test):
        if(self._status == 'quantized'):
            y_pred = self._model_predict(np.array(X_test))
            return y_pred
        else:
            print('Info: Please convert the model into a TreeLUT model first!')

    def verilog(self):
        if(self._folder_created == False):
            if os.path.exists(self._dir_path):
                shutil.rmtree(self._dir_path)
            os.makedirs(self._dir_path)
            self._folder_created = True

        if(self._status == 'quantized'):
            os.makedirs(os.path.join(self._dir_path, 'verilog'))
            unique_features = self._extract_unique_features()
            self._verilog_unique_features(unique_features)
            trees_bit_length = self._verilog_trees(unique_features)
            self._verilog_adder(trees_bit_length)
            self._verilog_ports()
            self._verilog_myreg()
            if(self._n_classes == 1):
                print(f"Info: Threshold for Binary Classification: {int(self._threshold)}")    
        else:
            print('Info: Please convert the model into a TreeLUT model first!')
            
    def testbench(self, X_test, y_test):
        if(self._folder_created == False):
            if os.path.exists(self._dir_path):
                shutil.rmtree(self._dir_path)
            os.makedirs(self._dir_path)
            self._folder_created = True
        
        if(self._status == 'quantized'):
            os.makedirs(os.path.join(self._dir_path, 'testbench'))
            self._verilog_testbench(np.array(X_test), np.array(y_test))
        else:
            print('Info: Please convert the model into a TreeLUT model first!')
        
    def _bitwidth(self, N):
        if N >= 0:
            return int(np.floor(np.log2(N)) + 1) if N > 0 else 1
    
    def _node_regex(self, s):
        patterns = {'index': r'(\d+):', 'feature': r'\[([^\<]+)', 'threshold': r'<([-\d\.]+)', 'yes_value': r'yes=(\d+)', 'no_value': r'no=(\d+)'}
        values = {key: re.search(pattern, s).group(1) for key, pattern in patterns.items() if re.search(pattern, s)}  
        result = {
            'index': int(values.get('index', 0)),
            'feature': values.get('feature', ''),
            'threshold': float(values.get('threshold', 0.0)),
            'yes_value': int(values.get('yes_value', 0)),
            'no_value': int(values.get('no_value', 0))
        }
        return result

    def _leaf_regex(self, s):
        pattern = r'(\d+):leaf=([-\d\.eE]+)'
        match = re.search(pattern, s)
        if match:
            index = int(match.group(1))
            value = float(match.group(2))
            return index, value

    def _format_model(self):
        formatted_trees = []
        for tree_str in self._xgb_model.get_booster().get_dump():
            tree_lines = tree_str.strip().split('\n')
            nodes = {}
            for line in tree_lines:
                if 'leaf' in line:
                    node_id, value = self._leaf_regex(line)
                    nodes[node_id] = {'type': 'leaf', 'value': value}
                else:
                    line_info = self._node_regex(line)
                    feature_idx = self._features.index(line_info['feature'])
                    nodes[line_info['index']] = {'type': 'split', 'feature': feature_idx, 'threshold': line_info['threshold'], 'no': line_info['no_value'], 'yes': line_info['yes_value']}
            formatted_trees.append(nodes)
        self._treelut_model = formatted_trees

    def _minmax_trees(self):
        min_trees = np.zeros((len(self._treelut_model),))
        max_range = -np.inf
        for i, tree in enumerate(self._treelut_model):
            all_values = np.array([d['value'] for d in tree.values() if 'value' in d])
            min_trees[i] = all_values.min()
            max_range = max((all_values.max()-all_values.min()), max_range)
        return([min_trees, max_range])

    def _quantize_model(self):
        trees_bias = np.zeros((self._n_classes, self._n_trees_per_class))
        [min_trees, max_range] = self._minmax_trees()
        scale = (2 ** self._w_tree - 1) / (max_range)
            
        for i in range(len(self._treelut_model)):
            trees_bias[i%self._n_classes][int(np.floor(i/self._n_classes))] = min_trees[i]
            for j in range(len(self._treelut_model[i])):
                if(self._treelut_model[i][j]['type'] == 'leaf'):
                    self._treelut_model[i][j]['value'] = int(np.round((self._treelut_model[i][j]['value'] - min_trees[i])*scale))
                else:
                    self._treelut_model[i][j]['threshold'] = int(np.ceil(self._treelut_model[i][j]['threshold']))
                    yes_node = self._treelut_model[i][j]['yes']
                    no_node = self._treelut_model[i][j]['no']
                    self._treelut_model[i][yes_node]['parent_node'] = j
                    self._treelut_model[i][yes_node]['parent_yesno'] = 'yes'
                    self._treelut_model[i][no_node]['parent_node'] = j
                    self._treelut_model[i][no_node]['parent_yesno'] = 'no'
        
        if(self._n_classes == 1):
            self._threshold = int(np.round(((-np.sum(min_trees))-self._base_score)*scale))
            
        self._classes_bias = trees_bias.sum(axis=1)
        self._classes_bias = self._classes_bias - self._classes_bias.min()
        self._classes_bias = (np.round((self._classes_bias) * scale)).astype(int)

    def _single_tree_predict(self, tree, X):
        node_id = np.zeros((X.shape[0], )).astype(int)
        is_leaf = (tree[node_id, 1] == 1).astype(bool)
        while np.any(is_leaf == False):
            feature = tree[node_id, 2]
            threshold = tree[node_id, 3]
            comparison = X[np.arange(X.shape[0]), feature] < (threshold-(1e-8))
            node_id[(~is_leaf) & (comparison == True)] = tree[node_id[(~is_leaf) & (comparison == True)], 4]
            node_id[(~is_leaf) & (comparison == False)] = tree[node_id[(~is_leaf) & (comparison == False)], 5]
            is_leaf = (tree[node_id, 1] == 1)
        return tree[node_id, 3]
    
    def _model_predict(self, X_test):
        treelut_model_numpy = []
        for i in range(len(self._treelut_model)):
            tree_numpy = np.zeros((len(self._treelut_model[i]), 6)).astype(int)
            for j in range(len(self._treelut_model[i])):
                if(self._treelut_model[i][j]['type'] == 'leaf'):
                    tree_numpy[j, 0] = j # node id
                    tree_numpy[j, 1] = 1 # node type (leaf)
                    tree_numpy[j, 3] = self._treelut_model[i][j]['value'] # leaf value
                else:
                    tree_numpy[j, 0] = j # node id
                    tree_numpy[j, 1] = 0 # node type (split)
                    tree_numpy[j, 2] = self._treelut_model[i][j]['feature'] # feature
                    tree_numpy[j, 3] = self._treelut_model[i][j]['threshold'] # threshold
                    tree_numpy[j, 4] = self._treelut_model[i][j]['yes'] # yes val
                    tree_numpy[j, 5] = self._treelut_model[i][j]['no'] # no val
            treelut_model_numpy.append(tree_numpy)
            
        predictions = np.zeros((X_test.shape[0], self._n_classes))
        for i in range(self._n_classes):
            predictions[:, i] = self._classes_bias[i]

        for i, tree in enumerate(treelut_model_numpy):
            class_number = i%self._n_classes
            predictions[:, class_number] += self._single_tree_predict(tree, X_test)
                
        if(self._n_classes == 1):
            return predictions >= (self._threshold)
        else:
            return np.argmax(predictions, axis=1)

    def _extract_unique_features(self):
        uf_list = []
        for tree in self._treelut_model:
            uf_list.extend([(d['feature'], d['threshold']) for d in tree.values() if 'feature' in d])
        uf_list = list(set(uf_list))
        uf_list.sort(key=lambda x: (x[0], x[1]))
        return uf_list

    def _verilog_unique_features(self, unique_features):
        file = open(os.path.join(self._dir_path, 'verilog/TreeLUT.v'), 'w')
        file.write(f"wire [{len(unique_features)-1}:0] binary_features;\n")
        if(self._pipeline[0] != 0):
            file.write(f"wire [{len(unique_features)-1}:0] binary_features_reg;\n")
            file.write(f"myreg #(.DataWidth({len(unique_features)})) feature_reg (.clk(clk), .data_in(binary_features), .data_out(binary_features_reg));\n")
        file.write("\n")
        for i, uf in enumerate((unique_features)):
            feature_index = int(uf[0])
            feature_input_bits = f"{(feature_index+1)*self._w_feature-1}:{feature_index*self._w_feature}"
            file.write(f"assign binary_features[{i}] = (i[{feature_input_bits}] < ({self._w_feature}'d{int(uf[1])}))? 1'b1 : 1'b0;\n")
        file.close()
        
    def _verilog_trees(self, unique_features):
        trees_bit_length = np.zeros((self._n_classes, self._n_trees_per_class), dtype=int)
        
        for tree_idx, tree in enumerate(self._treelut_model):
            class_number = tree_idx%self._n_classes
            tree_number = int(np.floor(tree_idx/self._n_classes))
            
            n_output_level = 2**self._w_tree
            path_encoded = np.empty(n_output_level, dtype=object)
            path_encoded[:] = [''*n_output_level]
            
            all_leaves = []
            for idx in range(len(tree)):
                if(tree[idx]['type'] == 'leaf'):
                    all_leaves.extend([(idx, tree[idx]['value'])])                

            for leaf in all_leaves:
                one_path = '('
                node_idx = leaf[0]
                while(node_idx != 0):
                    parent_idx = tree[node_idx]['parent_node']
                    parent_yesno = tree[node_idx]['parent_yesno']
                    parent_feature = (tree[parent_idx]['feature'], tree[parent_idx]['threshold'])
                    unique_feature_idx = unique_features.index(parent_feature)
                    if parent_yesno == 'yes':
                        one_path += f"i[{unique_feature_idx}] & "
                    else:
                        one_path += f"(~i[{unique_feature_idx}]) & "
                    node_idx = parent_idx
                one_path = one_path[:-3] + ") | "
                path_encoded[int(leaf[1])] += one_path

            path_max = 2**self._w_tree-1
            for path in reversed(path_encoded):
                if(path == ''):
                    path_max -= 1
                else:
                    break
            
            path_output_bit = self._bitwidth(path_max)
            trees_bit_length[class_number, tree_number] = path_output_bit          
            
            tree_name = f"class{class_number}_tree{tree_number}"
            file = open(os.path.join(self._dir_path, f"verilog/{tree_name}.v"), 'w')
            file.write(f"module {tree_name}(input wire [{len(unique_features)-1}:0] i, output wire [{path_output_bit-1}:0] o);\n\n")
            file.write(f"reg [{path_output_bit-1}:0] o_reg;\nassign o = o_reg;\n\nalways @(*) begin\n")
                
            have_else = 0    
            for path_idx, path in enumerate(path_encoded):
                if((path_max) == path_idx):
                    break
                if(path != ''):
                    if(have_else):
                        file.write("\telse if")
                    else:
                        file.write("\tif")
                        have_else = 1
                    file.write(f"({path[:-3]})\n\t\to_reg = {path_idx};\n")
            if(have_else):
                file.write(f"\telse\n\t\to_reg = {path_max};\n")   
            else:
                file.write(f"\to_reg = {path_max};\n")   
            file.write("end\n\nendmodule\n")
            file.close()

        file = open(os.path.join(self._dir_path, 'verilog/TreeLUT.v'), 'a')
        file.write(f"\n\nwire [{trees_bit_length.sum().sum()-1}:0] trees_output;\n")
        if(self._pipeline[1] != 0):
            file.write(f"wire [{trees_bit_length.sum().sum()-1}:0] trees_output_reg;\n")
            file.write(f"myreg #(.DataWidth({trees_bit_length.sum().sum()})) trees_reg (.clk(clk), .data_in(trees_output), .data_out(trees_output_reg));\n")
        file.write("\n")   
        idx = 0
        for i in range(self._n_classes):
            for j in range(self._n_trees_per_class):
                tree_name = f"class{i}_tree{j}"
                if(self._pipeline[0] != 0):
                    feature_name = "binary_features_reg"
                else:
                    feature_name = "binary_features"
                file.write(f"{tree_name} {tree_name}_inst(.i({feature_name}), .o(trees_output[{idx+trees_bit_length[i][j]-1}:{idx}]));\n")
                idx += trees_bit_length[i][j]
        file.close()

        return trees_bit_length
          
    def _verilog_addertree(self, class_number, class_bias, trees_bit_length):
        file = open(os.path.join(self._dir_path, f"verilog/class{class_number}_adder.v"), 'w')
        file.write(f"module class{class_number}_adder(")
        if(self._pipeline[2] != 0):
            file.write(f"input wire clk, ")
        file.write(f"input wire [{trees_bit_length.sum()-1}:0] i, output wire [{self._sum_bit_length-1}:0] o);\n\n")
        file.write(f"reg[{self._sum_bit_length-1}:0] o_reg;\nassign o = o_reg;\n\n")
        
        total_n_stage = np.ceil(np.log2(self._n_trees_per_class + (class_bias != 0)))
        pipeline_position = np.ceil(total_n_stage/(self._pipeline[2]+1))
        
        operands_name = []
        operands_bits = []
        if(class_bias != 0):
            n_bias_bits = self._bitwidth(class_bias)
            operands_name.append(f"{n_bias_bits}'d{class_bias}")
            operands_bits.append(n_bias_bits)
        
        for i in range(self._n_trees_per_class):
            operands_name.append(f"i[{trees_bit_length[0:i+1].sum()-1}:{trees_bit_length[0:i].sum()}]")
            operands_bits.append(trees_bit_length[i])
            
        n_stage = 0
        critical_path = 1
        while(True):
            regs_str = ""
            adds_str = ""
            new_operands_name = []
            new_operands_bits = []
            for i in range(int(len(operands_name)/2)):
                n_bits_addition = max(operands_bits[2*i], operands_bits[2*i+1]) + 1
                regs_str += f"reg [{n_bits_addition-1}:0] stage{n_stage}_adder{i};\n"
                adds_str += f"\tstage{n_stage}_adder{i} = {operands_name[2*i]} + {operands_name[2*i+1]};\n"
                new_operands_name.append(f"stage{n_stage}_adder{i}")
                new_operands_bits.append(n_bits_addition)
            
            if(len(operands_name)%2 != 0):
                n_bits_addition = operands_bits[-1]
                regs_str += f"reg [{n_bits_addition-1}:0] stage{n_stage}_adder{int(len(operands_name)/2)};\n"
                adds_str += f"\tstage{n_stage}_adder{int(len(operands_name)/2)} = {operands_name[-1]};\n"
                new_operands_name.append(f"stage{n_stage}_adder{int(len(operands_name)/2)}")
                new_operands_bits.append(n_bits_addition)
            
            file.write(regs_str)
            if(self._pipeline[2] != 0 and critical_path == pipeline_position):
                file.write(f"\nalways @(posedge clk) begin\n")
            else:
                file.write(f"\nalways @(*) begin\n")
            file.write(adds_str)
            file.write("end\n\n")
            
            operands_name = new_operands_name
            operands_bits = new_operands_bits
            
            if(len(new_operands_name) == 2):
                break
            
            critical_path += 1
            n_stage += 1
            
        file.write(f"\nalways @(*) begin\n")
        file.write(f"o_reg = {new_operands_name[0]} + {new_operands_name[1]};\n")
        file.write("end\n\nendmodule\n")
        
        file.close()
        
    def _verilog_adder(self, trees_bit_length):
        self._sum_bit_length = self._bitwidth(np.max(self._classes_bias + (np.power(2, trees_bit_length)-1).sum(axis=1)))    
        trees_bit_length = trees_bit_length.reshape(-1, order='C')
        
        file = open(os.path.join(self._dir_path, 'verilog/TreeLUT.v'), 'a')
        file.write("\n")
        if(self._pipeline[1] != 0):
            tree_output_name = "trees_output_reg"
        else:
            tree_output_name = "trees_output"
            
        if(self._pipeline[2] != 0):
            adder_clk = ".clk(clk), "
        else:
            adder_clk = ""
            
        for i in range(self._n_classes):
            trees_range = f"{trees_bit_length[0:(i+1)*self._n_trees_per_class].sum()-1}:{trees_bit_length[0:(i)*self._n_trees_per_class].sum()}"
            file.write(f"class{i}_adder class{i}_adder_inst({adder_clk}.i({tree_output_name}[{trees_range}]), .o(o[{(i+1)*self._sum_bit_length-1}:{i*self._sum_bit_length}]));\n") 
            self._verilog_addertree(i, self._classes_bias[i], trees_bit_length[i*self._n_trees_per_class:(i+1)*self._n_trees_per_class])
        file.close()
    
    def _verilog_ports(self):
        file_path = os.path.join(self._dir_path, 'verilog/TreeLUT.v')
        if(self._pipeline[0] == 0 and self._pipeline[1] == 0 and self._pipeline[2] == 0):
            first_line = f"module TreeLUT(input wire [{self._n_features*self._w_feature-1}:0] i, output wire [{self._n_classes*self._sum_bit_length-1}:0] o);\n\n"
        else:
            first_line = f"module TreeLUT(input wire clk, input wire [{self._n_features*self._w_feature-1}:0] i, output wire [{self._n_classes*self._sum_bit_length-1}:0] o);\n\n"
        last_line = f"\nendmodule\n"
        
        with open(file_path, 'r') as file:
            content = file.read()
            
        with open(file_path, 'w') as file:
            file.write(first_line + content + last_line)
            
    def _verilog_myreg(self):
        if(self._pipeline[0] != 0 or self._pipeline[1] != 0 or self._pipeline[2] != 0):
            file = open(os.path.join(self._dir_path, f"verilog/myreg.v"), 'w')
            file.write(f"module myreg #(parameter DataWidth=16) (input wire clk, input [DataWidth-1:0] data_in, output reg [DataWidth-1:0] data_out);\n")
            file.write(f"\nalways@(posedge clk) begin\n")
            file.write(f"\tdata_out <= data_in;\n")
            file.write(f"end\n\nendmodule\n")
        
    def _verilog_testbench(self, X_test, y_test):
        with open(os.path.join(self._dir_path, 'testbench/X_test.mem'), 'w') as f:
            for key in X_test:
                for value in np.flip(key):
                    x_val = value
                    binary_string = format(int(x_val), f'0{self._w_feature}b')
                    f.write(binary_string)
                f.write("\n")
                
        with open(os.path.join(self._dir_path, 'testbench/y_test.mem'), 'w') as f:
            n_output_bits = self._bitwidth(self._n_classes-1)
            for value in y_test:
                binary_string = format(int(value), f'0{n_output_bits}b')
                f.write(binary_string)
                f.write("\n")
                
        with open(os.path.join(self._dir_path, 'testbench/testbench.v'), 'w') as f:
            n_output_bits = self._bitwidth(self._n_classes-1)
            f.write(f"`timescale 1ns/1ps\n\nmodule sim ();\n\nreg [{self._n_features*self._w_feature-1}:0] x_test [0:{len(y_test)-1}];\n")
            f.write(f"reg [{n_output_bits-1}:0] y_test [0:{len(y_test)-1}];\n\n")
            
            f.write(f"reg [{self._n_features*self._w_feature-1}:0] treelut_input;\n")
            f.write(f"wire [{self._n_classes*self._sum_bit_length-1}:0] treelut_output;\n\n")
            
            f.write(f"wire [{n_output_bits-1}:0] out_predicted;\n")
            f.write(f"reg [{n_output_bits-1}:0] out_expected;\n\n")
            
            f.write(f"integer i, j;\n")
            
            n_address_bits = self._bitwidth(len(y_test)-1)
            f.write(f"reg [{n_address_bits-1}:0] result_false;\n")
            f.write(f"reg [{n_address_bits-1}:0] result_true;\n")
            
            if(self._pipeline[0] == 0 and self._pipeline[1] == 0 and self._pipeline[2] == 0):
                f.write(f"\nTreeLUT TreeLUT_inst(treelut_input, treelut_output);\n")
            else:
                f.write(f"reg clk;\n\n")
                f.write(f"TreeLUT TreeLUT_inst(clk, treelut_input, treelut_output);\n")
                
            f.write(f"argmax argmax_inst(treelut_output, out_predicted);\n\n")
            
            f.write(f"initial begin\n\t\n\tresult_false <= 0; result_true <= 0;\n\t$readmemb(\"X_test.mem\", x_test); $readmemb(\"y_test.mem\", y_test);\nend\n\n")
            
            if(self._pipeline[0] != 0 or self._pipeline[1] != 0 or self._pipeline[2] != 0):
                f.write(f"initial begin\n\tclk <= 0;\n\tforever #5 clk <= ~clk;\nend\n\n")
    
            f.write(f"initial begin\n\t#105\n\tfor (i = 0; i <= {len(y_test)}; i = i + 1) begin\n\t\t#10\n\t\ttreelut_input <= x_test[i];\n\tend\nend\n\n")
            
            f.write(f"initial begin\n\t#{105+10*(self._pipeline[0]+self._pipeline[1]+self._pipeline[2])}\n")
            f.write(f"\tfor (j = 0; j <= {len(y_test)}; j = j + 1) begin\n\t\t#10\n\t\tout_expected <= y_test[j];\n")
            f.write(f"\t\tif(out_predicted == out_expected)\n\t\t\tresult_true <= result_true + 1;\n\t\telse\n\t\t\tresult_false <= result_false+1;\n\tend\n")
            f.write(f"\t$finish;\nend\nendmodule")

            f.write(f"\n\nmodule argmax(")
            f.write(f"input wire [{self._n_classes*self._sum_bit_length-1}:0] i, ")
            f.write(f"output reg [{n_output_bits-1}:0] o);\n\n")

            f.write(f"wire [{self._sum_bit_length-1}:0] sum [0:{self._n_classes-1}];\n")
            for i in range(self._n_classes):
                f.write(f"assign sum[{i}] = i[{(i+1)*(self._sum_bit_length)-1}:{i*(self._sum_bit_length)}];\n")

            f.write("always @(*) begin")
            if(self._n_classes == 1):
                f.write(f"\n\tif(sum[0] >= {int(self._threshold)})\n\t\to = 1;\n\telse\n\t\to = 0;\n")
            else:
                for i in range(self._n_classes):
                    if(i == 0):
                        f.write(f"\n\tif(")
                    elif(i == self._n_classes-1):
                        f.write(f"\telse\n\t\to = {n_output_bits}'d{i};\n")
                        break
                    else:
                        f.write(f"\telse if(")
                    comprator_str = ''
                    for j in range(self._n_classes):
                        if(i >= j):
                            continue
                        comprator_str += f"(sum[{i}] >= sum[{j}]) & "
                    f.write(comprator_str[:-3] + f")\n\t\to = {n_output_bits}'d{i};\n")

            f.write("end\n\nendmodule\n")
            f.close()
