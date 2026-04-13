# file_path=/data/yanfei/SES/local_wiki_server/wiki-18-e5-index
# index_file=$file_path/e5_Flat.index
# corpus_file=$file_path/wiki-18.jsonl
# retriever_name=e5
# retriever_path=intfloat/e5-base-v2

# CUDA_VISIBLE_DEVICES=7 python retrieval_server.py   --index_path $index_file \
#                                                     --corpus_path $corpus_file \
#                                                     --retrieval_topk 10 \
#                                                     --retrieval_method $retriever_name \
#                                                     --retrieval_model_path $retriever_path \
#                                                     --faiss_gpu True

CUDA_VISIBLE_DEVICES=7 python retrieval_server.py