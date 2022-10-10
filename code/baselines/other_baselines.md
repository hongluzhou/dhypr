## Air
#### Task Link Prediction, Air
- **NERD**: ```./NERD -train ../../data/air/General_Directed_Link_Prediction/fold_0/train_edges_weights.txt -output1 ../../NERD_embeddings/air/General_Directed_Link_Prediction/fold_0/hub.txt -output2 ../../NERD_embeddings/air/General_Directed_Link_Prediction/fold_0/auth.txt -binary 0 -size 16 -walkSize 3 -negative 0 -samples 10 -rho 0.025 -threads 20 -joint 1 -inputvertex 0```
- **ATP**: ```./main_atp.py --dag ../../data/air/General_Directed_Link_Prediction/fold_0/train_edges_DAG.txt --rank 32 --strategy ln --id_mapping False --using_GPU False --dense_M False --using_SVD False```
- **APP**: ```java PPREmbedding ../../data/air/General_Directed_Link_Prediction/fold_0/train_edges.txt layer_size 32 alpha 0.0025f starting_alpha 0.05f jump_factor 0.15f MAX_EXP 5 magic 100 neg 5```

## Blog
#### Task Link Prediction, Blog
- **NERD**: ```./NERD -train ../../data/blog/General_Directed_Link_Prediction/fold_0/train_edges_weights.txt -output1 ../../NERD_embeddings/blog/General_Directed_Link_Prediction/fold_0/hub.txt -output2 ../../NERD_embeddings/blog/General_Directed_Link_Prediction/fold_0/auth.txt -binary 0 -size 16 -walkSize 3 -negative 0 -samples 10 -rho 0.025 -threads 20 -joint 1 -inputvertex 0```
- **ATP**: ```./main_atp.py --dag ../../data/blog/General_Directed_Link_Prediction/fold_0/train_edges_DAG.txt --rank 32 --strategy ln --id_mapping False --using_GPU False --dense_M False --using_SVD False```
- **APP**: ```java PPREmbedding ../../data/blog/General_Directed_Link_Prediction/fold_0/train_edges.txt layer_size 32 alpha 0.0025f starting_alpha 0.05f jump_factor 0.15f MAX_EXP 5 magic 100 neg 5```

## Cora
#### Task Link Prediction, Cora
- **NERD**: ```./NERD -train ../../data/cora/General_Directed_Link_Prediction/fold_0/train_edges_weights.txt -output1 ../../NERD_embeddings/cora/General_Directed_Link_Prediction/fold_0/hub.txt -output2 ../../NERD_embeddings/cora/General_Directed_Link_Prediction/fold_0/auth.txt -binary 0 -size 16 -walkSize 3 -negative 0 -samples 1 -rho 0.025 -threads 20 -joint 1 -inputvertex 0```
- **ATP**: ```./main_atp.py --dag ../../data/cora/General_Directed_Link_Prediction/fold_0/train_edges_DAG.txt --rank 32 --strategy ln --id_mapping False --using_GPU False --dense_M False --using_SVD False```
- **APP**: ```java PPREmbedding ../../data/cora/General_Directed_Link_Prediction/fold_0/train_edges.txt layer_size 32 alpha 0.0025f starting_alpha 0.05f jump_factor 0.15f MAX_EXP 5 magic 100 neg 5```

## Survey
#### Task Link Prediction, Survey
- **NERD**: ```./NERD -train ../../data/survey/General_Directed_Link_Prediction/fold_0/train_edges_weights.txt -output1 ../../NERD_embeddings/survey/General_Directed_Link_Prediction/fold_0/hub.txt -output2 ../../NERD_embeddings/survey/General_Directed_Link_Prediction/fold_0/auth.txt -binary 0 -size 16 -walkSize 3 -negative 0 -samples 10 -rho 0.025 -threads 20 -joint 1 -inputvertex 0```
- **ATP**: ```./main_atp.py --dag ../../data/survey/General_Directed_Link_Prediction/fold_0/train_edges_DAG.txt --rank 32 --strategy ln --id_mapping False --using_GPU False --dense_M False --using_SVD False```
- **APP**: ```java PPREmbedding ../../data/survey/General_Directed_Link_Prediction/fold_0/train_edges.txt layer_size 32 alpha 0.0025f starting_alpha 0.05f jump_factor 0.15f MAX_EXP 5 magic 100 neg 5```

## DBLP
#### Task Link Prediction, DBLP
- **NERD**: ```./NERD -train ../../data/dblp/General_Directed_Link_Prediction/fold_0/train_edges_weights.txt -output1 ../../NERD_embeddings/dblp/General_Directed_Link_Prediction/fold_0/hub.txt -output2 ../../NERD_embeddings/dblp/General_Directed_Link_Prediction/fold_0/auth.txt -binary 0 -size 16 -walkSize 3 -negative 0 -samples 10 -rho 0.025 -threads 20 -joint 1 -inputvertex 0```
- **ATP**: ```./main_atp.py --dag ../../data/dblp/General_Directed_Link_Prediction/fold_0/train_edges_DAG.txt --rank 32 --strategy ln --id_mapping False --using_GPU False --dense_M False --using_SVD False```
- **APP**: ```java PPREmbedding ../../data/dblp/General_Directed_Link_Prediction/fold_0/train_edges.txt layer_size 32 alpha 0.0025f starting_alpha 0.05f jump_factor 0.15f MAX_EXP 5 magic 100 neg 5```
