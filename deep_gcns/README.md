# ogbn-proteins

We simply apply a random partition to generate batches for both mini-batch training and test. We set the number of partitions to be 10 for training and 5 for test, and we set the batch size to 1 subgraph.  We initialize the features of nodes through aggregating the features of their connected edges by a Sum (Add) aggregation.
## Default 
    --use_gpu False 
    --cluster_number 10 
    --valid_cluster_number 5 
    --aggr add 	#options: [mean, max, add]
    --block plain 	#options: [plain, res, res+]
    --conv gen
    --gcn_aggr max 	#options: [max, mean, add, softmax, softmax_sg, softmax_sum, power, power_sum]
    --num_layers 3
    --conv_encode_edge False
    --use_one_hot_encoding False
    --mlp_layers 2
    --norm layer
    --hidden_channels 64
    --epochs 1000
    --lr 0.001
    --dropout 0.0
    --num_evals 1

## PlainGCN 
	python main.py --use_gpu --conv_encode_edge --use_one_hot_encoding --num_layers 28 --gcn_aggr mean

## ResGCN 
	python main.py --use_gpu --conv_encode_edge --use_one_hot_encoding --num_layers 28 --block res --gcn_aggr mean

    
