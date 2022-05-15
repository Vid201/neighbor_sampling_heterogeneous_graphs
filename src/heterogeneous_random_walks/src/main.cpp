#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <unordered_map>
#include <string>
#include <sstream>
#include <vector>
#include <utility>

namespace py = pybind11;

std::pair<torch::Tensor, torch::Tensor> heterogeneous_random_walk(const std::vector<std::string> &node_types, const std::vector<std::string> &edge_types, std::unordered_map<std::string, torch::Tensor> &rowptr, std::unordered_map<std::string, torch::Tensor> &col, const torch::Tensor &start, const std::vector<std::string> &start_types, const int64_t &walk_length) {
    std::unordered_map<std::string, unsigned int> node_types_dict; // node type string -> node type unique id
    std::unordered_map<std::string, unsigned int> edge_types_dict; // edge type string -> edge type unique id
    std::unordered_map<unsigned int, std::string> rev_edge_types_dict; // edge type unique id -> edge type string
    std::unordered_map<unsigned int, unsigned int> edge_types_ingoing; // edge type unique id -> node type unique id (ingoing node)
    std::unordered_map<unsigned int, unsigned int> node_types_outgoing_number; // node type unique id -> number of unique outgoing edge types
    std::unordered_map<unsigned int, std::vector<unsigned int>> node_types_outgoing_list; // node type unique id -> list of unique outoing edge types
    std::unordered_map<unsigned int, std::unordered_map<unsigned int, std::pair<int64_t*, int64_t*>>> adj_ptr; // node type unique id -> edge type unique id -> adjacency matrix (rowptr, col)
    
    unsigned int counter = 0;
    
    for (auto node_type : node_types) {
        node_types_dict[node_type] = counter++;
    }
    
    std::string a, b, c;
    unsigned int from, to;
    counter = 0;
    
    for (auto edge_type : edge_types) {
        std::stringstream ss(edge_type);
        ss >> a >> b >> c;
        edge_types_dict[edge_type] = counter;
        rev_edge_types_dict[counter] = edge_type;
        edge_types_ingoing[counter] = node_types_dict[c];
        counter += 1;
    }
    
    for (auto item : rowptr) {
        std::stringstream ss(item.first);
        ss >> a >> b >> c;
        from = node_types_dict[a];
        to = edge_types_dict[item.first];
        
        if (node_types_outgoing_list.find(from) == node_types_outgoing_list.end()) {
            node_types_outgoing_number[from] = 0;
            node_types_outgoing_list[from] = std::vector<unsigned int>();
            adj_ptr[from] = std::unordered_map<unsigned int, std::pair<int64_t*, int64_t*>>();
        }
        
        node_types_outgoing_number[from] += 1;
        node_types_outgoing_list[from].push_back(to);
        adj_ptr[from][to] = std::make_pair(item.second.data_ptr<int64_t>(), col[item.first].data_ptr<int64_t>());
    }
    
    int64_t *start_ptr = start.data_ptr<int64_t>();
    
    auto rws_node_types = torch::empty({start.size(0), walk_length + 1}, start.options());
    auto rws_edge_types = torch::empty({start.size(0), walk_length}, start.options());
    int64_t *rws_node_types_ptr = rws_node_types.data_ptr<int64_t>();
    int64_t *rws_edge_types_ptr = rws_edge_types.data_ptr<int64_t>();
    int64_t grain_size = at::internal::GRAIN_SIZE / walk_length;
    
    auto rand_edge_type = torch::rand({start.numel(), walk_length});
    auto rand_edge_type_ptr = rand_edge_type.data_ptr<float>();
    auto rand_neighbor = torch::rand({start.numel(), walk_length});
    auto rand_neighbor_ptr = rand_neighbor.data_ptr<float>();
    
    at::parallel_for(0, start.numel(), grain_size, [&](int64_t begin, int64_t end) {
        for (auto n = begin; n < end; n++) {
            int64_t curr_node = start_ptr[n], curr_type = node_types_dict[start_types[n]], curr_edge, edge_type, row_start, row_end, idx;
            rws_node_types_ptr[n * (walk_length + 1)] = curr_node;

            for (auto l = 0; l < walk_length; l++) {
                // select random edge type (uniformly)
                edge_type = node_types_outgoing_list[curr_type][rand_edge_type_ptr[n * walk_length + l] * node_types_outgoing_number[curr_type]];
                row_start = adj_ptr[curr_type][edge_type].first[curr_node], row_end = adj_ptr[curr_type][edge_type].first[curr_node + 1];
                
                if (row_end - row_start > 0) {
                    // select random neighbor from that edge type (uniformly)
                    idx = rand_neighbor_ptr[n * walk_length + l] * (row_end - row_start);
                    curr_edge = row_start + idx;
                    curr_node = adj_ptr[curr_type][edge_type].second[curr_edge];
                    curr_type = edge_types_ingoing[edge_type];
                } else {
                    edge_type = -1;
                }
                
                rws_node_types_ptr[n * (walk_length + 1) + (l + 1)] = curr_node;
                rws_edge_types_ptr[n * walk_length + l] = edge_type;
            }
        }
    });

    return std::make_pair(rws_node_types, rws_edge_types);
}

PYBIND11_MODULE(random_walks, handle) {
    handle.doc() = "This is heterogeneous random walks module.";
    handle.def("heterogeneous_random_walk", &heterogeneous_random_walk);
}